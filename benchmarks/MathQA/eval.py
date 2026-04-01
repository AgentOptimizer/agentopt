"""
MathQA Self-Reflective Agent Benchmark — Bedrock edition.

Two-agent workflow: an answer agent solves multiple-choice math problems and
a critic agent verifies the solution. The loop repeats until the critic
approves or a maximum number of iterations is reached.

2-tuple benchmark: answer x critic (10x10 = 100 combos with all models).

Dataset: https://huggingface.co/datasets/allenai/math_qa

Usage:
    python -m benchmarks.MathQA.eval --limit 20
    python -m benchmarks.MathQA.eval --all-models --limit 40 --no-cache --parallel
    python -m benchmarks.MathQA.eval --mode raw --limit 20
    python -m benchmarks.MathQA.eval --selector bayesian_optimization --limit 40
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGSMITH_TRACING", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langgraph.graph import END, StateGraph

from benchmarks.MathQA.tools import TOOLS as CALC_TOOLS
from benchmarks.MathQA.tools import calculator

from benchmarks.common import (
    BEDROCK_PRICES,
    DEFAULT_MODELS,
    add_common_cli_args,
    build_selector_kwargs,
    display_name,
    extract_text_content,
    get_selectors,
    make_llm,
    resolve_models,
    supports_tool_calling,
)

logger = logging.getLogger(__name__)

SELECTORS = get_selectors()

ANSWER_SYSTEM_PROMPT = (
    "You are a math problem solver. You will be given a multiple-choice math "
    "question with options labeled a through e.\n\n"
    "You have tools available: a calculator for arithmetic, a unit converter, "
    "and an equation solver. Use them for any computation you are not 100% "
    "confident about.\n\n"
    "Solve the problem step by step and output your final answer as a single "
    "letter on the last line, prefixed with 'ANSWER: '. For example: ANSWER: c\n\n"
    "If you have received feedback from a reviewer indicating your previous "
    "answer was wrong, carefully reconsider and correct your approach."
)

ANSWER_SYSTEM_PROMPT_NO_TOOLS = (
    "You are a math problem solver. You will be given a multiple-choice math "
    "question with options labeled a through e.\n\n"
    "Solve the problem step by step, performing all calculations yourself.\n\n"
    "Output your final answer as a single letter on the last line, prefixed "
    "with 'ANSWER: '. For example: ANSWER: c\n\n"
    "If you have received feedback from a reviewer indicating your previous "
    "answer was wrong, carefully reconsider and correct your approach."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a math answer reviewer. You will see a math question, its "
    "multiple-choice options, and a proposed solution.\n\n"
    "Verify whether the proposed answer is correct by solving the problem "
    "yourself.\n\n"
    "If the answer is correct, respond with exactly: CORRECT\n"
    "If the answer is incorrect, respond with: INCORRECT: <brief explanation "
    "of the error and a hint toward the right approach>\n\n"
    "Do not reveal the correct answer letter directly."
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_math_qa(
    train_split: float = 0.8,
    max_samples: int = 200,
) -> tuple[list, list]:
    """Load allenai/math_qa from HuggingFace and split into train/test."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        )

    ds = hf_load_dataset("allenai/math_qa", split="test", trust_remote_code=True)

    tasks = []
    for row in ds:
        question_text = f"{row['Problem']}\n\nOptions: {row['options']}"
        input_data = {
            "messages": [{"role": "user", "content": question_text}],
        }
        expected = row["correct"].strip().lower()
        tasks.append((input_data, expected))
        if len(tasks) >= max_samples:
            break

    split_idx = int(len(tasks) * train_split)
    return tasks[:split_idx], tasks[split_idx:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_messages(messages: List[Any]) -> List[Any]:
    return list(convert_to_messages(messages))


def _is_system_message(message: Any) -> bool:
    if isinstance(message, dict):
        return message.get("role") == "system"
    if hasattr(message, "type"):
        return message.type == "system"
    return False


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        if "final" in result:
            return extract_text_content(result["final"])
        if "messages" in result and result["messages"]:
            last = result["messages"][-1]
            if isinstance(last, dict):
                return extract_text_content(last.get("content", ""))
            if hasattr(last, "content"):
                return extract_text_content(last.content)
            return extract_text_content(last)
    return extract_text_content(result)


def eval_fn(expected: str, actual: Any) -> bool:
    """Compare agent's final answer letter to the correct option."""
    text = _extract_text(actual).lower()
    expected_letter = expected.strip().lower()

    match = re.search(r"answer:\s*([a-e])", text)
    if match:
        return match.group(1) == expected_letter

    match = re.search(r"\b([a-e])\b", text[-50:])
    if match:
        return match.group(1) == expected_letter

    return False


# ---------------------------------------------------------------------------
# Tool call helper
# ---------------------------------------------------------------------------


def _run_answer_with_tools(llm, messages, use_tools=True):
    """Run answer agent with optional tool calling loop."""
    if use_tools:
        model_with_tools = llm.bind_tools(CALC_TOOLS)
        response = model_with_tools.invoke(messages)
        messages = messages + [response]

        tools_by_name = {t.name: t for t in CALC_TOOLS}
        while getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                tool_fn = tools_by_name.get(tc["name"], calculator)
                tool_result = tool_fn.invoke(tc["args"])
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                )
            response = model_with_tools.invoke(messages)
            messages.append(response)
    else:
        response = llm.invoke(messages)
        messages = messages + [response]

    return messages


# ---------------------------------------------------------------------------
# Raw mode agent_fn factory
# ---------------------------------------------------------------------------


def _mathqa_agent_fn_raw(models: Dict[str, Any], max_iterations: int = 3):
    """Factory: raw mode agent_fn for MathQA.

    models: {"answer": "<model>", "critic": "<model>"}
    Returns a callable: (input_data) -> result_dict
    """
    answer_spec = models["answer"]
    critic_spec = models["critic"]

    def run(input_data: dict) -> dict:
        answer_llm = make_llm(answer_spec) if isinstance(answer_spec, str) else answer_spec
        critic_llm = make_llm(critic_spec) if isinstance(critic_spec, str) else critic_spec
        use_tools = supports_tool_calling(answer_spec) if isinstance(answer_spec, str) else True

        answer_sys = ANSWER_SYSTEM_PROMPT if use_tools else ANSWER_SYSTEM_PROMPT_NO_TOOLS
        messages = _normalize_messages(list(input_data.get("messages", [])))
        if not messages or not _is_system_message(messages[0]):
            messages = [SystemMessage(content=answer_sys)] + messages

        final = ""
        for iteration in range(max_iterations):
            # Answer agent
            messages = _run_answer_with_tools(answer_llm, messages, use_tools)

            # Critic agent
            review_prompt = HumanMessage(
                content=(
                    "Review the previous answer. Verify it by solving the "
                    "problem yourself. Reply CORRECT if the answer is right, "
                    "or INCORRECT: <feedback> if it is wrong."
                )
            )
            critic_messages = (
                [SystemMessage(content=CRITIC_SYSTEM_PROMPT)]
                + messages[1:]
                + [review_prompt]
            )
            critic_response = critic_llm.invoke(critic_messages)
            messages.append(critic_response)

            content = extract_text_content(getattr(critic_response, "content", str(critic_response)))
            if "CORRECT" in content.upper() and "INCORRECT" not in content.upper():
                break

        # Extract final answer
        for msg in reversed(messages):
            raw = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            content = extract_text_content(raw)
            if re.search(r"answer:\s*[a-e]", content.lower()):
                final = content
                break
        if not final and messages:
            last = messages[-1]
            final = extract_text_content(getattr(last, "content", str(last)))

        return {"messages": messages, "final": final.strip(), "iteration": iteration + 1}

    return run


# ---------------------------------------------------------------------------
# LangGraph mode agent_fn factory
# ---------------------------------------------------------------------------


class ReflectionState(TypedDict):
    messages: List[Any]
    final: str
    iteration: int


def _mathqa_agent_fn_langgraph(models: Dict[str, Any], max_iterations: int = 3):
    """Factory: LangGraph mode agent_fn for MathQA.

    models: {"answer": "<model>", "critic": "<model>"}
    Returns a callable: (input_data) -> result_dict
    """
    answer_spec = models["answer"]
    critic_spec = models["critic"]

    def run(input_data: dict) -> dict:
        answer_llm = make_llm(answer_spec) if isinstance(answer_spec, str) else answer_spec
        critic_llm = make_llm(critic_spec) if isinstance(critic_spec, str) else critic_spec
        use_tools = supports_tool_calling(answer_spec) if isinstance(answer_spec, str) else True
        answer_sys = ANSWER_SYSTEM_PROMPT if use_tools else ANSWER_SYSTEM_PROMPT_NO_TOOLS
        max_iter = max_iterations

        def add_system(state: ReflectionState) -> ReflectionState:
            messages = _normalize_messages(list(state.get("messages", [])))
            if messages and _is_system_message(messages[0]):
                return {"messages": messages, "final": "", "iteration": 0}
            return {
                "messages": [SystemMessage(content=answer_sys)] + messages,
                "final": "",
                "iteration": 0,
            }

        def call_answer_agent(state: ReflectionState) -> ReflectionState:
            messages = _normalize_messages(list(state.get("messages", [])))
            messages = _run_answer_with_tools(answer_llm, messages, use_tools)
            return {"messages": messages, "final": "", "iteration": state.get("iteration", 0)}

        def call_critic_agent(state: ReflectionState) -> ReflectionState:
            messages = _normalize_messages(list(state.get("messages", [])))
            review_prompt = HumanMessage(
                content="Review the previous answer. Reply CORRECT or INCORRECT: <feedback>."
            )
            critic_messages = (
                [SystemMessage(content=CRITIC_SYSTEM_PROMPT)]
                + messages[1:]
                + [review_prompt]
            )
            response = critic_llm.invoke(critic_messages)
            return {"messages": messages + [response], "final": "", "iteration": state.get("iteration", 0)}

        def check_critic(state: ReflectionState) -> ReflectionState:
            return {
                "messages": list(state.get("messages", [])),
                "final": "",
                "iteration": state.get("iteration", 0) + 1,
            }

        def route_after_critic(state: ReflectionState) -> str:
            messages = list(state.get("messages", []))
            iteration = state.get("iteration", 0)
            if messages:
                last = messages[-1]
                raw = getattr(last, "content", str(last)) if not isinstance(last, dict) else last.get("content", "")
                content = extract_text_content(raw)
                if "CORRECT" in content.upper() and "INCORRECT" not in content.upper():
                    return "finalize"
            if iteration >= max_iter:
                return "finalize"
            return "call_answer_agent"

        def finalize(state: ReflectionState) -> ReflectionState:
            messages = list(state.get("messages", []))
            final = ""
            for msg in reversed(messages):
                raw = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
                content = extract_text_content(raw)
                if re.search(r"answer:\s*[a-e]", content.lower()):
                    final = content
                    break
            if not final and messages:
                last = messages[-1]
                raw = getattr(last, "content", str(last)) if not isinstance(last, dict) else last.get("content", "")
                final = extract_text_content(raw)
            return {"messages": messages, "final": final.strip(), "iteration": state.get("iteration", 0)}

        graph = StateGraph(ReflectionState)
        graph.add_node("add_system", add_system)
        graph.add_node("call_answer_agent", call_answer_agent)
        graph.add_node("call_critic_agent", call_critic_agent)
        graph.add_node("check_critic", check_critic)
        graph.add_node("finalize", finalize)

        graph.set_entry_point("add_system")
        graph.add_edge("add_system", "call_answer_agent")
        graph.add_edge("call_answer_agent", "call_critic_agent")
        graph.add_edge("call_critic_agent", "check_critic")
        graph.add_conditional_edges(
            "check_critic",
            route_after_critic,
            {"finalize": "finalize", "call_answer_agent": "call_answer_agent"},
        )
        graph.add_edge("finalize", END)
        compiled = graph.compile()

        return compiled.invoke(input_data)

    return run


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results, title="Model Performance", save_path=None):
    plt.figure(figsize=(10, 6))
    accuracies = [r.accuracy for r in results]
    latencies = [r.latency_seconds for r in results]
    names = [r.model_name for r in results]

    plt.scatter(latencies, accuracies)
    for name, lat, acc in zip(names, latencies, accuracies):
        plt.annotate(display_name(name), (lat, acc))

    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Self-Reflective MathQA Benchmark with AgentOpt (Bedrock)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.MathQA.eval --limit 20\n"
            "  python -m benchmarks.MathQA.eval --all-models --limit 40 --no-cache --parallel\n"
            "  python -m benchmarks.MathQA.eval --mode raw --limit 20\n"
        ),
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--train-split", type=float, default=0.8,
        help="Fraction of data for model selection (default: 0.8)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Maximum dataset samples (default: 200)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Max self-reflection iterations (default: 3)",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    models = resolve_models(args)
    mode_label = "Raw API" if args.mode == "raw" else "LangGraph"

    print("=" * 60)
    print(f"Self-Reflective MathQA Benchmark  [{mode_label}]")
    print("=" * 60)

    # Dataset
    print("\n[1] Loading math_qa dataset from HuggingFace...")
    train_data, test_data = load_math_qa(
        train_split=args.train_split,
        max_samples=args.max_samples,
    )
    if args.limit:
        train_data = train_data[: args.limit]
    print(f"  Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # Agent factory
    print(f"\n[2] Setting up self-reflective agents ({mode_label})...")
    print(f"    Models: {models}")
    print(f"    Search space: {len(models)}x{len(models)} = {len(models)**2} combos")

    if args.mode == "raw":
        def agent_fn(m):
            return _mathqa_agent_fn_raw(m, max_iterations=args.max_iterations)
    else:
        def agent_fn(m):
            return _mathqa_agent_fn_langgraph(m, max_iterations=args.max_iterations)

    # Model selection
    print(f"\n[3] Running model selection ({args.selector})...")
    selector_kwargs = build_selector_kwargs(args)

    from agentopt import LLMTracker

    SelectorCls = SELECTORS[args.selector]
    selector = SelectorCls(
        agent=agent_fn,
        models={"answer": models, "critic": models},
        eval_fn=eval_fn,
        dataset=train_data,
        model_prices=BEDROCK_PRICES,
        **selector_kwargs,
    )

    if args.no_cache:
        selector._tracker.stop()
        tracker = LLMTracker(cache=False)
        selector._tracker = tracker
        tracker.start()

    start = time.time()
    results = selector.select_best(
        parallel=args.parallel,
        per_combo=getattr(args, 'per_combo', False),
    )
    elapsed = time.time() - start

    # Results
    print(f"\n{'=' * 60}")
    print(f"MathQA Results ({mode_label})")
    print(f"{'=' * 60}")
    print(f"Train samples: {len(train_data)} | Selector: {args.selector} | Time: {elapsed:.1f}s")
    print()

    results.print_summary()

    if args.csv:
        results.to_csv(args.csv)
        print(f"Results saved to: {args.csv}")

    if args.plot:
        plot_results(results, f"Self-Reflective MathQA — {mode_label}", args.plot)


if __name__ == "__main__":
    main()
