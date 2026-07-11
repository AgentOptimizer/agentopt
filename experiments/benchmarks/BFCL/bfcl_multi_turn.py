"""
BFCL v3 Multi-Turn Benchmark — download, evaluation, and model selection.

Multi-turn function calling against simulated backends (GorillaFileSystem,
TradingBot, TravelAPI, etc.).  Evaluation is execution-based: compare final
backend state, not function call strings.  Uses AgentOpt to compare models
via any selector.

Dataset: https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html

Usage:
    # Quick test (5 samples, default model)
    python benchmarks/BFCL/bfcl_multi_turn.py --limit 5

    # Full benchmark with specific models
    python benchmarks/BFCL/bfcl_multi_turn.py --models 'bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0'

    # Use a different selector
    python benchmarks/BFCL/bfcl_multi_turn.py --limit 20 --selector hill_climbing

    # Run all models
    python benchmarks/BFCL/bfcl_multi_turn.py --limit 10 --all-models

    # Disable caching
    python benchmarks/BFCL/bfcl_multi_turn.py --limit 10 --no-cache

    # Save results
    python benchmarks/BFCL/bfcl_multi_turn.py --limit 50 --csv results.csv
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from pydantic import Field, create_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.common import get_selectors

logger = logging.getLogger(__name__)

SELECTORS = get_selectors()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCHMARK_DIR / "backend"
FUNC_DOC_DIR = BENCHMARK_DIR / "func_doc"
DATA_DIR = BENCHMARK_DIR / "data"

CLASS_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

STATELESS_CLASSES = ["MathAPI"]
MAX_STEPS_PER_TURN = 20

SYSTEM_MESSAGE = (
    "You are interacting with a set of simulated backend APIs. "
    "All tools work on all file types regardless of extension — "
    "for example, grep, sort, cat, and diff all work on .pdf files. "
    "Always use the provided tools to complete tasks. "
    "Do not refuse to call a tool based on assumptions about file formats. "
    "Operations like mkdir, mv, cp, and touch only work on items in the "
    "current directory — use cd to navigate first before operating on files."
)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
HF_DATA_BASE = (
    "https://huggingface.co/datasets/gorilla-llm/"
    "Berkeley-Function-Calling-Leaderboard/resolve/main"
)


def download_bfcl(limit: int | None = None) -> Path:
    """Download BFCL v3 multi-turn data from HuggingFace."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_file = DATA_DIR / "BFCL_v3_multi_turn_base.json"
    answer_dir = DATA_DIR / "possible_answer"
    answer_dir.mkdir(parents=True, exist_ok=True)
    answer_file = answer_dir / "BFCL_v3_multi_turn_base.json"

    if not test_file.exists():
        _download_file(
            f"{HF_DATA_BASE}/BFCL_v3_multi_turn_base.json",
            test_file,
        )

    if not answer_file.exists():
        _download_file(
            f"{HF_DATA_BASE}/possible_answer/BFCL_v3_multi_turn_base.json",
            answer_file,
        )

    n_test = sum(1 for _ in open(test_file))
    n_ans = sum(1 for _ in open(answer_file))
    print(f"Dataset: {n_test} test samples, {n_ans} answers")
    return DATA_DIR


def _download_file(url: str, dest: Path):
    """Download a file with retries."""
    print(f"Downloading {url} ...")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"  Saved to {dest}")
            return
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/3 after {wait}s ({e})")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_bfcl_dataset(
    limit: int | None = None,
) -> list[tuple[dict, list[list[str]]]]:
    """Load BFCL v3 multi-turn as (input_data, ground_truth) tuples."""
    data_dir = download_bfcl(limit=limit)
    test_file = data_dir / "BFCL_v3_multi_turn_base.json"
    answer_file = data_dir / "possible_answer" / "BFCL_v3_multi_turn_base.json"

    tests = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))

    ground_truths = {}
    with open(answer_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                ground_truths[entry["id"]] = entry["ground_truth"]

    dataset = []
    for test in tests:
        gt = ground_truths.get(test["id"], [])
        dataset.append((test, gt))
        if limit and len(dataset) >= limit:
            break

    return dataset


# ---------------------------------------------------------------------------
# Backend instantiation
# ---------------------------------------------------------------------------
_backend_module_cache: Dict[str, Any] = {}


def _load_backend_module(module_name: str):
    if module_name in _backend_module_cache:
        return _backend_module_cache[module_name]
    backend_dir = str(BACKEND_DIR)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        module_name, str(BACKEND_DIR / f"{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _backend_module_cache[module_name] = module
    return module


def create_backend_instances(
    initial_config: dict,
    involved_classes: List[str],
    long_context: bool = False,
) -> Dict[str, Any]:
    instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_MAPPING.get(class_name)
        if not module_name:
            continue
        module = _load_backend_module(module_name)
        cls = getattr(module, class_name)
        instance = cls()
        if class_name not in STATELESS_CLASSES:
            class_config = initial_config.get(class_name, {})
            instance._load_scenario(copy.deepcopy(class_config), long_context=long_context)
        instances[class_name] = instance
    return instances


# ---------------------------------------------------------------------------
# Tool building (BFCL func_doc → LangGraph StructuredTools)
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "string": str, "integer": int, "number": float, "float": float,
    "boolean": bool, "array": list, "object": dict, "dict": dict,
}


def _make_args_schema(fn_name: str, parameters: dict):
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    fields = {}
    for pname, pschema in props.items():
        ptype = _TYPE_MAP.get(pschema.get("type", "string"), Any)
        desc = pschema.get("description", "")
        safe_pname = pname.lstrip("_") if pname.startswith("_") else pname
        if pname in required:
            fields[safe_pname] = (ptype, Field(description=desc))
        else:
            default = pschema.get("default", None)
            if isinstance(default, str):
                if default == "None":
                    default = None
                elif default.lower() == "true":
                    default = True
                elif default.lower() == "false":
                    default = False
            fields[safe_pname] = (Optional[ptype], Field(default=default, description=desc))
    if not fields:
        fields["placeholder"] = (Optional[str], Field(default=None, description="No parameters needed"))
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", fn_name) + "_Schema"
    return create_model(safe_name, **fields)


def load_func_docs(involved_classes: List[str]) -> List[dict]:
    all_funcs = []
    for class_name in involved_classes:
        module_name = CLASS_FILE_MAPPING.get(class_name)
        if not module_name:
            continue
        doc_file = FUNC_DOC_DIR / f"{module_name}.json"
        if not doc_file.exists():
            continue
        with open(doc_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_funcs.append(json.loads(line))
    return all_funcs


def build_tools(func_docs: List[dict], instances: Dict[str, Any]) -> List[StructuredTool]:
    method_to_instance = {}
    for class_name, instance in instances.items():
        for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            method_to_instance[method_name] = instance

    tools = []
    for func_doc in func_docs:
        fn_name = func_doc["name"]
        desc = func_doc.get("description", f"Call {fn_name}")
        params = func_doc.get("parameters", {})
        args_schema = _make_args_schema(fn_name, params)

        def make_fn(name: str):
            def _fn(**kwargs) -> str:
                inst = method_to_instance.get(name)
                if not inst:
                    return f"Error: function '{name}' not found in backend"
                method = getattr(inst, name)
                try:
                    filtered = {k: v for k, v in kwargs.items() if v is not None}
                    result = method(**filtered)
                    if isinstance(result, dict):
                        try:
                            return json.dumps(result)
                        except (TypeError, ValueError):
                            return str(result)
                    return str(result) if result is not None else "Success"
                except Exception as e:
                    return f"Error: {e}"
            return _fn

        tool = StructuredTool.from_function(
            func=make_fn(fn_name), name=fn_name,
            description=desc, args_schema=args_schema,
        )
        tools.append(tool)
    return tools


# ---------------------------------------------------------------------------
# LLM factory & helpers — use shared common.py (OpenRouter)
# ---------------------------------------------------------------------------


from benchmarks.common import make_llm, display_name as _display_name, extract_text_content, supports_tool_calling


# ---------------------------------------------------------------------------
# LangGraph agent (FC mode — native tool calling)
# ---------------------------------------------------------------------------
def build_agent_graph(llm_with_tools: Any, tools: List[StructuredTool]):
    """Build a LangGraph ReAct agent: agent → tools → agent loop."""
    def call_model(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# Prompting mode (non-FC models like DeepSeek-R1)
# ---------------------------------------------------------------------------
def _func_docs_to_prompt(func_docs: List[dict]) -> str:
    """Convert func_doc JSON definitions into a text description for the system prompt."""
    lines = ["You have access to the following functions. To call a function, "
             "output EXACTLY one or more lines in this format:\n"
             "  function_name(arg1=value1, arg2=value2)\n\n"
             "Use Python literal syntax for values (strings in quotes, numbers bare, "
             "lists with brackets, etc.). Output ONLY function calls, no explanation.\n"
             "If you have no function to call, output: NO_CALL\n\n"
             "Available functions:\n"]
    for fd in func_docs:
        name = fd["name"]
        desc = fd.get("description", "")
        params = fd.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        param_parts = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "string")
            pdesc = pschema.get("description", "")
            req = "(required)" if pname in required else "(optional)"
            param_parts.append(f"    - {pname} ({ptype}, {req}): {pdesc}")
        params_str = "\n".join(param_parts) if param_parts else "    (no parameters)"
        lines.append(f"  {name}: {desc}\n  Parameters:\n{params_str}\n")
    return "\n".join(lines)


def _parse_function_calls(text: str) -> List[str]:
    """Parse function call strings from model text output.

    Looks for lines matching: function_name(...)
    """
    calls = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line == "NO_CALL":
            continue
        # Match function_name(...) pattern
        if re.match(r"^\w+\s*\(.*\)\s*$", line):
            calls.append(line)
    return calls


def _build_method_map(instances: Dict[str, Any]) -> Dict[str, Any]:
    """Build a method_name -> instance map from backend instances."""
    method_map = {}
    for class_name, instance in instances.items():
        for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            method_map[method_name] = instance
    return method_map


def _auto_quote_args(args_str: str) -> str:
    """Auto-quote bare-word positional arguments in a function call arg string.

    Some models (e.g. Qwen3 Next 80B A3B) output function calls with unquoted
    string args like ``cd(document)`` instead of ``cd("document")``.  This
    helper turns bare identifiers into quoted strings so ``eval()`` succeeds.
    Keyword args that are already quoted or numeric are left unchanged.
    """
    # Strip outer parens: "(document, temp)" -> "document, temp"
    inner = args_str.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    if not inner.strip():
        return args_str  # no args

    parts = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            parts.append(part)
            continue
        # keyword arg: name=value
        if "=" in part:
            key, val = part.split("=", 1)
            val = val.strip()
            # Already quoted, numeric, boolean, None, or list/dict — leave as-is
            if (val.startswith(("'", '"', "[", "{", "("))
                    or val in ("True", "False", "None")
                    or re.match(r"^-?[\d.]+$", val)):
                parts.append(part)
            else:
                parts.append(f'{key.strip()}="{val}"')
        else:
            # Positional arg
            if (part.startswith(("'", '"', "[", "{", "("))
                    or part in ("True", "False", "None")
                    or re.match(r"^-?[\d.]+$", part)):
                parts.append(part)
            else:
                parts.append(f'"{part}"')
    return "(" + ", ".join(parts) + ")"


def _execute_and_format(call_str: str, method_map: Dict[str, Any]) -> str:
    """Execute a parsed function call string and return the result as text."""
    match = re.match(r"(\w+)\s*\(", call_str)
    if not match:
        return f"Error: cannot parse '{call_str}'"
    fn_name = match.group(1)
    instance = method_map.get(fn_name)
    if not instance:
        return f"Error: function '{fn_name}' not found"
    method = getattr(instance, fn_name, None)
    if not method:
        return f"Error: method '{fn_name}' not found"
    args_str = call_str[len(fn_name):]
    try:
        result = eval(f"method{args_str}", {"method": method})
        if isinstance(result, dict):
            try:
                return json.dumps(result)
            except (TypeError, ValueError):
                return str(result)
        return str(result) if result is not None else "Success"
    except NameError:
        # Bare-word args — auto-quote and retry
        quoted_args = _auto_quote_args(args_str)
        try:
            result = eval(f"method{quoted_args}", {"method": method})
            if isinstance(result, dict):
                try:
                    return json.dumps(result)
                except (TypeError, ValueError):
                    return str(result)
            return str(result) if result is not None else "Success"
        except Exception as e2:
            return f"Error: {e2}"
    except Exception as e:
        return f"Error: {e}"


def run_sample_prompting(
    test_entry: dict,
    llm: Any,
) -> Tuple[Dict[str, Any], float, Optional[str], List[List[Tuple[str, dict]]]]:
    """Run one BFCL multi-turn sample in prompting mode (no native tool calling).

    Tools are described in the system prompt. The model outputs function calls
    as plain text, which we parse and execute against the backends.

    Returns (instances, latency_s, error, tool_calls_per_turn).
    """
    involved_classes = test_entry["involved_classes"]
    initial_config = test_entry["initial_config"]
    turns = test_entry["question"]

    instances = create_backend_instances(initial_config, involved_classes)
    func_docs = load_func_docs(involved_classes)
    method_map = _build_method_map(instances)

    tool_prompt = _func_docs_to_prompt(func_docs)
    system_msg = SYSTEM_MESSAGE + "\n\n" + tool_prompt

    messages = [SystemMessage(content=system_msg)]
    all_tool_calls: List[List[Tuple[str, dict]]] = []

    t0 = time.time()
    for turn_idx, turn_messages in enumerate(turns):
        for msg in turn_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))

        turn_calls: List[Tuple[str, dict]] = []
        step_exceeded = False

        # Let the model call functions in a loop (up to MAX_STEPS_PER_TURN)
        for step in range(MAX_STEPS_PER_TURN):
            try:
                response = llm.invoke(messages)
                raw_content = response.content if hasattr(response, "content") else str(response)
                # Some models (DeepSeek-R1) return content as a list of blocks
                if isinstance(raw_content, list):
                    text_parts = []
                    for block in raw_content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict):
                            if "text" in block:
                                text_parts.append(block["text"])
                            elif "reasoningContent" in block:
                                rc = block["reasoningContent"]
                                if isinstance(rc, dict) and "reasoningText" in rc:
                                    rt = rc["reasoningText"]
                                    if isinstance(rt, dict) and "text" in rt:
                                        text_parts.append(rt["text"])
                                    elif isinstance(rt, str):
                                        text_parts.append(rt)
                    text = "\n".join(text_parts)
                else:
                    text = str(raw_content)
                messages.append(AIMessage(content=text))
            except Exception as e:
                return instances, time.time() - t0, f"Turn {turn_idx} step {step} error: {e}", all_tool_calls

            # Parse function calls from the output
            calls = _parse_function_calls(text)
            if not calls:
                # No function calls = end of turn
                break

            # Track each parsed call as (name, args)
            for call_str in calls:
                parsed_name, parsed_args = _parse_call_to_tuple(call_str)
                turn_calls.append((parsed_name, parsed_args))

            # Execute each call and feed results back
            results = []
            for call_str in calls:
                result = _execute_and_format(call_str, method_map)
                results.append(f"{call_str} -> {result}")

            feedback = "Function results:\n" + "\n".join(results)
            messages.append(HumanMessage(content=feedback))
        else:
            # for-loop completed without break → hit MAX_STEPS_PER_TURN
            step_exceeded = True

        all_tool_calls.append(turn_calls)

        if step_exceeded:
            return (
                instances, time.time() - t0,
                f"Turn {turn_idx}: exceeded {MAX_STEPS_PER_TURN} step limit",
                all_tool_calls,
            )

    return instances, time.time() - t0, None, all_tool_calls


# ---------------------------------------------------------------------------
# Run a single multi-turn sample (auto-detects FC vs prompting mode)
# ---------------------------------------------------------------------------
def _is_non_fc_model(model_str: str) -> bool:
    """Check if a model needs prompting mode (uses common.py's NO_TOOL_CALLING_MODELS)."""
    return not supports_tool_calling(model_str)


def run_sample(
    test_entry: dict,
    llm: Any,
    prompting_mode: bool = False,
) -> Tuple[Dict[str, Any], float, Optional[str], List[List[Tuple[str, dict]]]]:
    """Run one BFCL multi-turn sample.

    Returns (instances, latency_s, error, tool_calls_per_turn).
    Token tracking is handled automatically by agentproxy's botocore interception.
    If prompting_mode=True, uses text-based function calling instead of native tools.
    """
    if prompting_mode:
        return run_sample_prompting(test_entry, llm)

    involved_classes = test_entry["involved_classes"]
    initial_config = test_entry["initial_config"]
    turns = test_entry["question"]

    instances = create_backend_instances(initial_config, involved_classes)
    func_docs = load_func_docs(involved_classes)
    tools = build_tools(func_docs, instances)

    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception as e:
        return {}, 0.0, f"LLM bind_tools error: {e}", []

    app = build_agent_graph(llm_with_tools, tools)
    messages = [SystemMessage(content=SYSTEM_MESSAGE)]
    all_tool_calls: List[List[Tuple[str, dict]]] = []

    t0 = time.time()
    for turn_idx, turn_messages in enumerate(turns):
        for msg in turn_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))

        old_count = len(messages)
        try:
            result = app.invoke(
                {"messages": messages},
                # Tight limit: 20 agent→tool cycles + 1 final agent response
                config={"recursion_limit": MAX_STEPS_PER_TURN * 2 + 1},
            )
            messages = result["messages"]
        except Exception as e:
            err_str = str(e)
            if "recursion" in err_str.lower():
                return (
                    instances, time.time() - t0,
                    f"Turn {turn_idx}: exceeded {MAX_STEPS_PER_TURN} step limit",
                    all_tool_calls,
                )
            return instances, time.time() - t0, f"Turn {turn_idx} error: {e}", all_tool_calls

        # Extract tool calls from new messages added during this turn
        turn_calls: List[Tuple[str, dict]] = []
        step_count = 0
        for msg in messages[old_count:]:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                step_count += 1
                for tc in msg.tool_calls:
                    turn_calls.append((tc["name"], tc.get("args", {})))

        if step_count > MAX_STEPS_PER_TURN:
            return (
                instances, time.time() - t0,
                f"Turn {turn_idx}: {step_count} steps exceeded limit of {MAX_STEPS_PER_TURN}",
                all_tool_calls,
            )
        all_tool_calls.append(turn_calls)

    return instances, time.time() - t0, None, all_tool_calls


# ---------------------------------------------------------------------------
# Raw API baseline (no framework — matches BFCL official evaluation)
# ---------------------------------------------------------------------------
def run_sample_raw(
    test_entry: dict,
    llm: Any,
    prompting_mode: bool = False,
) -> Tuple[Dict[str, Any], float, Optional[str], List[List[Tuple[str, dict]]]]:
    """Run one BFCL multi-turn sample using raw API calls (no LangGraph).

    This matches BFCL's official evaluation methodology:
    - bind_tools() on the LLM directly
    - Manual while loop: send messages → execute tool calls → append results → repeat
    - No StateGraph, no ToolNode, no framework overhead

    For prompting-mode models, delegates to run_sample_prompting() (already framework-free).

    Returns (instances, latency_s, error, tool_calls_per_turn).
    """
    if prompting_mode:
        return run_sample_prompting(test_entry, llm)

    involved_classes = test_entry["involved_classes"]
    initial_config = test_entry["initial_config"]
    turns = test_entry["question"]

    instances = create_backend_instances(initial_config, involved_classes)
    func_docs = load_func_docs(involved_classes)
    tools = build_tools(func_docs, instances)

    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception as e:
        return {}, 0.0, f"LLM bind_tools error: {e}", []

    # Build a name→tool map for executing tool calls
    tool_map = {t.name: t for t in tools}

    messages: list = [SystemMessage(content=SYSTEM_MESSAGE)]
    all_tool_calls: List[List[Tuple[str, dict]]] = []

    t0 = time.time()
    for turn_idx, turn_messages in enumerate(turns):
        for msg in turn_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))

        turn_calls: List[Tuple[str, dict]] = []
        step_count = 0

        # Manual tool-calling loop (matches BFCL's official approach)
        while step_count < MAX_STEPS_PER_TURN:
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                return (
                    instances, time.time() - t0,
                    f"Turn {turn_idx} step {step_count} invoke error: {e}",
                    all_tool_calls,
                )

            messages.append(response)

            # Check if the model made any tool calls
            if not getattr(response, "tool_calls", None):
                # No tool calls — model is done with this turn
                break

            step_count += 1

            # Execute each tool call and append results
            from langchain_core.messages import ToolMessage

            for tc in response.tool_calls:
                fn_name = tc["name"]
                fn_args = tc.get("args", {})
                turn_calls.append((fn_name, fn_args))

                tool = tool_map.get(fn_name)
                if tool:
                    try:
                        result = tool.invoke(fn_args)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Error: function '{fn_name}' not found"

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
        else:
            # while-loop condition failed → exceeded step limit
            all_tool_calls.append(turn_calls)
            return (
                instances, time.time() - t0,
                f"Turn {turn_idx}: exceeded {MAX_STEPS_PER_TURN} step limit",
                all_tool_calls,
            )

        all_tool_calls.append(turn_calls)

    return instances, time.time() - t0, None, all_tool_calls


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_sample(
    test_entry: dict,
    ground_truth: list[list[str]],
    model_instances: Dict[str, Any],
    model_calls_per_turn: Optional[List[List[Tuple[str, dict]]]] = None,
) -> Tuple[bool, str]:
    """Evaluate with BOTH state-based AND response-based checks.

    A sample passes only if:
    1. State-based: final backend state matches ground truth state.
    2. Response-based: GT function calls appear as subsequence of model calls.
    """
    involved_classes = test_entry["involved_classes"]
    initial_config = test_entry["initial_config"]

    if not model_instances:
        return False, "No model instances (sample didn't run)"

    # --- State-based check ---
    gt_instances = create_backend_instances(initial_config, involved_classes)

    for turn_gt_calls in ground_truth:
        gt_method_map = {}
        for class_name, instance in gt_instances.items():
            for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
                if method_name.startswith("_"):
                    continue
                gt_method_map[method_name] = instance

        for call_str in turn_gt_calls:
            try:
                _execute_call_string(call_str, gt_method_map)
            except Exception as e:
                return False, f"Ground truth exec error: {e} on '{call_str}'"

    for class_name in involved_classes:
        model_inst = model_instances.get(class_name)
        gt_inst = gt_instances.get(class_name)
        if model_inst is None or gt_inst is None:
            return False, f"Missing instance for {class_name}"
        for attr_name in vars(gt_inst):
            if attr_name.startswith("_"):
                continue
            model_val = getattr(model_inst, attr_name, None)
            gt_val = getattr(gt_inst, attr_name, None)
            if model_val != gt_val:
                return False, (
                    f"State mismatch in {class_name}.{attr_name}: "
                    f"model={_truncate(str(model_val), 100)} vs "
                    f"gt={_truncate(str(gt_val), 100)}"
                )

    # --- Response-based check ---
    if model_calls_per_turn is not None:
        resp_ok, resp_reason = evaluate_response(ground_truth, model_calls_per_turn)
        if not resp_ok:
            return False, f"Response check failed: {resp_reason}"

    return True, ""


def _execute_call_string(call_str: str, method_map: Dict[str, Any]):
    match = re.match(r"(\w+)\s*\(", call_str)
    if not match:
        raise ValueError(f"Cannot parse call: {call_str}")
    fn_name = match.group(1)
    instance = method_map.get(fn_name)
    if not instance:
        raise ValueError(f"Function '{fn_name}' not found in method map")
    method = getattr(instance, fn_name)
    args_str = call_str[len(fn_name):]
    result = eval(f"method{args_str}", {"method": method})
    return result


def _truncate(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s


# ---------------------------------------------------------------------------
# Response-based evaluation helpers
# ---------------------------------------------------------------------------
def _parse_call_to_tuple(call_str: str) -> Tuple[str, dict]:
    """Parse a GT call string like 'func(a=1, b="x")' into ('func', {'a': 1, 'b': 'x'})."""
    match = re.match(r"(\w+)\s*\((.*)\)\s*$", call_str, re.DOTALL)
    if not match:
        return call_str.strip(), {}
    fn_name = match.group(1)
    args_str = match.group(2).strip()
    if not args_str:
        return fn_name, {}
    try:
        args = eval(f"dict({args_str})")
        return fn_name, args
    except Exception:
        # Fallback: name-only match if args can't be parsed
        return fn_name, {"_unparsed": args_str}


def _normalize_value(v: Any) -> Any:
    """Normalize a value for flexible comparison."""
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def _calls_match(
    gt_name: str, gt_args: dict, model_name: str, model_args: dict
) -> bool:
    """Check if a ground truth call matches a model call.

    Function names must match exactly.  GT args must be a subset of model args
    (the model may pass extra default arguments).
    """
    if gt_name != model_name:
        return False
    if "_unparsed" in gt_args:
        # Could not parse GT args — accept name-only match
        return True
    for k, v in gt_args.items():
        if k not in model_args:
            return False
        gt_v = _normalize_value(v)
        model_v = _normalize_value(model_args[k])
        if gt_v != model_v and str(gt_v) != str(model_v):
            return False
    return True


def _is_call_subsequence(
    gt_calls: List[Tuple[str, dict]],
    model_calls: List[Tuple[str, dict]],
) -> bool:
    """Check if gt_calls appears as an ordered subsequence of model_calls."""
    if not gt_calls:
        return True
    gt_idx = 0
    for model_name, model_args in model_calls:
        gt_name, gt_args = gt_calls[gt_idx]
        if _calls_match(gt_name, gt_args, model_name, model_args):
            gt_idx += 1
            if gt_idx >= len(gt_calls):
                return True
    return gt_idx >= len(gt_calls)


def evaluate_response(
    ground_truth: list[list[str]],
    model_calls_per_turn: List[List[Tuple[str, dict]]],
) -> Tuple[bool, str]:
    """Response-based evaluation: GT calls must be a subsequence of model calls per turn."""
    for turn_idx, gt_turn_calls in enumerate(ground_truth):
        gt_parsed = [_parse_call_to_tuple(c) for c in gt_turn_calls]
        model_turn_calls = (
            model_calls_per_turn[turn_idx]
            if turn_idx < len(model_calls_per_turn)
            else []
        )
        if not _is_call_subsequence(gt_parsed, model_turn_calls):
            gt_names = [name for name, _ in gt_parsed]
            model_names = [name for name, _ in model_turn_calls]
            return False, (
                f"Turn {turn_idx}: GT calls {gt_names} not found as "
                f"subsequence in model calls {model_names}"
            )
    return True, ""


# ---------------------------------------------------------------------------
# Eval function for AgentOpt selectors
# ---------------------------------------------------------------------------
def bfcl_eval_fn(expected: list[list[str]], actual: Any) -> float:
    """AgentOpt-compatible eval function.

    expected = ground truth (list of turns, each a list of call strings)
    actual = dict with 'instances', 'test_entry', and 'tool_calls_per_turn' from agent_fn

    A sample scores 1.0 only if BOTH state-based AND response-based checks pass.
    """
    if not isinstance(actual, dict) or "error" in actual:
        return 0.0
    test_entry = actual["test_entry"]
    instances = actual["instances"]
    tool_calls = actual.get("tool_calls_per_turn")
    is_correct, reason = evaluate_sample(test_entry, expected, instances, tool_calls)
    if not is_correct:
        logger.debug(f"Sample failed: {reason}")
    return 1.0 if is_correct else 0.0


# ---------------------------------------------------------------------------
# Agent factory for model selection (new agentopt API)
# ---------------------------------------------------------------------------
def bfcl_agent_fn(models: Dict[str, Any]):
    """Factory: takes a models dict and returns a callable that runs BFCL samples.

    models: {"agent": "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"}

    Returns a callable: (test_entry_dict) -> result_dict
    """
    model_spec = models["agent"]
    # Detect if this model needs prompting mode
    prompting = isinstance(model_spec, str) and _is_non_fc_model(model_spec)

    def run(input_data: dict) -> dict:
        # Create a fresh LLM per call — ChatBedrockConverse clients are NOT
        # thread-safe (sharing across concurrent threads causes HTTP response
        # state corruption).
        if isinstance(model_spec, str):
            llm = make_llm(model_spec)
        else:
            llm = model_spec

        instances, latency, err, tool_calls = run_sample(input_data, llm, prompting_mode=prompting)

        if err:
            return {"error": err, "test_entry": input_data, "instances": {}, "tool_calls_per_turn": tool_calls}
        return {
            "test_entry": input_data,
            "instances": instances,
            "latency": latency,
            "tool_calls_per_turn": tool_calls,
        }

    return run


def bfcl_agent_fn_raw(models: Dict[str, Any]):
    """Factory: raw API baseline (no LangGraph framework).

    Same interface as bfcl_agent_fn but uses run_sample_raw() instead.
    This matches BFCL's official evaluation methodology.
    """
    model_spec = models["agent"]
    prompting = isinstance(model_spec, str) and _is_non_fc_model(model_spec)

    def run(input_data: dict) -> dict:
        if isinstance(model_spec, str):
            llm = make_llm(model_spec)
        else:
            llm = model_spec

        instances, latency, err, tool_calls = run_sample_raw(
            input_data, llm, prompting_mode=prompting
        )

        if err:
            return {"error": err, "test_entry": input_data, "instances": {}, "tool_calls_per_turn": tool_calls}
        return {
            "test_entry": input_data,
            "instances": instances,
            "latency": latency,
            "tool_calls_per_turn": tool_calls,
        }

    return run


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results, title="BFCL Multi-Turn Model Performance", save_path=None):
    plt.figure(figsize=(10, 6))
    accuracies = [r.accuracy for r in results]
    latencies = [r.latency_seconds for r in results]
    names = [r.model_name for r in results]

    plt.scatter(latencies, accuracies, s=100, zorder=5)
    for name, lat, acc in zip(names, latencies, accuracies):
        plt.annotate(name, (lat, acc), textcoords="offset points", xytext=(8, 4), fontsize=9)

    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


# Pricing is handled by model_price.json via agentopt's built-in lookup


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def _get_agent_fn(mode: str):
    """Return the appropriate agent factory for the given mode."""
    if mode == "raw":
        return bfcl_agent_fn_raw
    else:  # "langgraph" (default)
        return bfcl_agent_fn


def run_direct_benchmark(
    models: list[str],
    dataset: list[tuple[dict, list[list[str]]]],
    parallel: bool = False,
    max_concurrent: int = 10,
    mode: str = "langgraph",
    no_cache: bool = False,
):
    """Run benchmark via BruteForceModelSelector using agent_fn factory."""
    from agentopt import LLMTracker

    agent_fn = _get_agent_fn(mode)
    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models={"agent": models},
        eval_fn=bfcl_eval_fn,
        dataset=dataset,
        # model_price.json handles pricing for OpenRouter model names
    )
    if no_cache:
        # Replace the auto-created tracker with one that has caching disabled
        selector._tracker.stop()
        tracker = LLMTracker(cache=False)
        selector._tracker = tracker
        tracker.start()
    results = selector.select_best(parallel=parallel, max_concurrent=max_concurrent)
    return results


def run_model_selection(
    dataset: list[tuple[dict, list[list[str]]]],
    models: list[str],
    selector_name: str = "brute_force",
    parallel: bool = False,
    max_concurrent: int = 10,
    selector_kwargs: dict | None = None,
    mode: str = "langgraph",
    no_cache: bool = False,
):
    """Run AgentOpt model selection on the dataset."""
    from agentopt import LLMTracker

    agent_fn = _get_agent_fn(mode)
    SelectorCls = SELECTORS[selector_name]
    base_kwargs = {
        "agent_fn": agent_fn,
        "models": {"agent": models},
        "eval_fn": bfcl_eval_fn,
        "dataset": dataset,
        # model_price.json handles pricing for OpenRouter model names
    }
    if selector_kwargs:
        base_kwargs.update(selector_kwargs)

    selector = SelectorCls(**base_kwargs)
    if no_cache:
        selector._tracker.stop()
        tracker = LLMTracker(cache=False)
        selector._tracker = tracker
        tracker.start()
    results = selector.select_best(parallel=parallel, max_concurrent=max_concurrent)
    print(f"\nBest: {results.get_best()}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Models are now passed via --models CLI arg (OpenRouter model IDs)


def main():
    parser = argparse.ArgumentParser(
        description="BFCL v3 Multi-Turn Benchmark with AgentOpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmarks/BFCL/bfcl_multi_turn.py --limit 5\n"
            "  python benchmarks/BFCL/bfcl_multi_turn.py --models 'bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0'\n"
            "  python benchmarks/BFCL/bfcl_multi_turn.py --limit 10 --selector hill_climbing\n"
            "  python benchmarks/BFCL/bfcl_multi_turn.py --all-models --limit 10\n"
        ),
    )
    parser.add_argument(
        "--models", nargs="+",
        default=None,
        help="Models to evaluate (OpenRouter model IDs, e.g. anthropic/claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run all default models (haiku, sonnet, nova-lite via Bedrock)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples to evaluate (default: all 199)",
    )
    parser.add_argument(
        "--selector", choices=sorted(SELECTORS.keys()), default=None,
        help="Model selection algorithm. If omitted, runs direct benchmark.",
    )
    parser.add_argument("--parallel", action="store_true", help="Parallel evaluation (asyncio)")
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        help="Max concurrent API calls per model (default: 10)",
    )
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV")
    parser.add_argument("--plot", type=str, default=None, help="Save plot to file")
    parser.add_argument(
        "--sample-fraction", type=float, default=0.5,
        help="Fraction for random_search (default: 0.5)",
    )
    parser.add_argument(
        "--reduction-factor", type=float, default=3.0,
        help="Reduction factor for hyperband (default: 3.0)",
    )
    parser.add_argument(
        "--mode", choices=["langgraph", "raw"], default="langgraph",
        help="Execution mode: 'langgraph' (LangGraph ReAct agent) or 'raw' (raw API baseline, no framework)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable agentproxy response cache")
    parser.add_argument("--verbose", action="store_true", help="Detailed logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    mode_label = "Raw API baseline" if args.mode == "raw" else "LangGraph ReAct agent"
    print("=" * 60)
    print(f"BFCL v3 Multi-Turn Benchmark  [{mode_label}]")
    print("=" * 60)

    # Dataset
    print("\n[1] Loading BFCL v3 multi-turn dataset ...")
    dataset = load_bfcl_dataset(limit=args.limit)
    print(f"    {len(dataset)} samples loaded")

    models = DEFAULT_MODELS if args.all_models else args.models

    if args.selector:
        # Model selection mode (specific selector algorithm)
        print(f"\n[2] Running model selection ({args.selector}) ...")
        selector_kwargs = {}
        if args.selector == "random_search":
            selector_kwargs["sample_fraction"] = args.sample_fraction
        if args.selector in ("hyperband", "hyperband_v2"):
            selector_kwargs["reduction_factor"] = args.reduction_factor

        results = run_model_selection(
            dataset, models,
            selector_name=args.selector,
            parallel=args.parallel,
            max_concurrent=args.max_concurrent,
            selector_kwargs=selector_kwargs,
            mode=args.mode,
            no_cache=args.no_cache,
        )
    else:
        # Direct benchmark mode (brute force over models)
        cache_label = " (cache OFF)" if args.no_cache else ""
        print(f"\n[2] Running benchmark ({mode_label}{cache_label}) ...")
        results = run_direct_benchmark(
            models, dataset,
            parallel=args.parallel,
            max_concurrent=args.max_concurrent,
            mode=args.mode,
            no_cache=args.no_cache,
        )

    print(f"\n{'=' * 60}")
    print("BFCL v3 Multi-Turn — Results")
    print(f"{'=' * 60}")
    results.print_summary()

    if args.csv:
        results.to_csv(args.csv)
        print(f"Results saved to {args.csv}")

    if args.plot:
        plot_results(results, "BFCL v3 Multi-Turn Benchmark", args.plot)


if __name__ == "__main__":
    main()
