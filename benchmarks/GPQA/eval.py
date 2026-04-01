"""
GPQA Diamond Benchmark — Bedrock edition.

Expert-level science multiple-choice QA (A/B/C/D). Evaluation is exact-match
on the answer letter. Uses AgentOpt to compare models via any selector.

1-tuple benchmark: single agent role, 10 model candidates.

Dataset: Wanfq/gpqa on HuggingFace (public mirror, no auth needed).

Usage:
    python -m benchmarks.GPQA.eval --limit 10
    python -m benchmarks.GPQA.eval --all-models --limit 40 --no-cache --parallel
    python -m benchmarks.GPQA.eval --models "Claude 3 Haiku" "DeepSeek R1" --limit 20
    python -m benchmarks.GPQA.eval --agent cot --selector hill_climbing
    python -m benchmarks.GPQA.eval --mode raw --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGSMITH_TRACING", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState

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
)

logger = logging.getLogger(__name__)

SELECTORS = get_selectors()

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

HF_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=Wanfq/gpqa&config=gpqa_diamond&split=train"
)

KEEP_COLUMNS = [
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
    "High-level domain",
    "Subdomain",
]

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def download_gpqa(data_dir: Path | str | None = None, limit: int | None = None) -> Path:
    """Download GPQA Diamond and save as JSONL."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dest = data_dir / "gpqa_diamond.jsonl"
    if dest.exists():
        n_existing = sum(1 for _ in open(dest))
        if limit is None or n_existing >= limit:
            print(f"Already downloaded: {dest} ({n_existing} rows)")
            return dest

    total = limit or 198
    batch_size = 100
    rows = []

    print(f"Downloading GPQA Diamond ({total} rows) ...")
    offset = 0
    while offset < total:
        length = min(batch_size, total - offset)
        url = f"{HF_API_URL}&offset={offset}&length={length}"
        req = urllib.request.Request(url, headers={"User-Agent": "agentopt/0.1"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                break
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                if attempt < 2:
                    wait = 2**attempt
                    print(f"  Retry {attempt + 1}/3 after {wait}s ({e})")
                    time.sleep(wait)
                else:
                    raise

        fetched = data.get("rows", [])
        if not fetched:
            break

        for item in fetched:
            row = item["row"]
            rows.append({k: row[k] for k in KEEP_COLUMNS if k in row})

        offset += len(fetched)
        print(f"  Fetched {offset}/{total} rows")

    with open(dest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rows to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_gpqa_dataset(
    data_dir: str | Path | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[tuple[dict[str, Any], str]]:
    """Load GPQA Diamond as ``(input_data, expected)`` tuples."""
    data_path = download_gpqa(data_dir=data_dir, limit=limit)

    rng = random.Random(seed)
    dataset: list[tuple[dict[str, Any], str]] = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if not row:
                continue

            question = row["Question"]
            correct = row["Correct Answer"]
            wrong = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]

            choices = [correct] + wrong
            rng.shuffle(choices)
            correct_idx = choices.index(correct)
            correct_letter = chr(ord("A") + correct_idx)

            letters = ["A", "B", "C", "D"]
            choices_text = "\n".join(
                f"  ({letters[i]}) {choices[i]}" for i in range(4)
            )
            formatted = (
                f"{question}\n\n"
                f"Choose one of the following:\n"
                f"{choices_text}\n\n"
                f"Answer with ONLY the letter (A, B, C, or D)."
            )

            input_data = {
                "input": formatted,
                "domain": row.get("High-level domain", ""),
                "subdomain": row.get("Subdomain", ""),
                "question_raw": question,
                "choices": dict(zip(letters, choices)),
            }
            dataset.append((input_data, correct_letter))

            if limit and len(dataset) >= limit:
                break

    return dataset


# ---------------------------------------------------------------------------
# Answer extraction & scoring
# ---------------------------------------------------------------------------


def extract_answer_letter(response: str) -> str | None:
    """Extract the answer letter (A/B/C/D) from a model response."""
    if not response:
        return None

    text = response.strip()

    clean = text.strip("().\" '")
    if clean.upper() in ("A", "B", "C", "D"):
        return clean.upper()

    patterns = [
        r"(?:my\s+)?(?:final\s+)?(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?",
        r"\\boxed\{([A-D])\}",
        r"\*\*([A-D])\*\*",
        r"\(([A-D])\)",
        r"^([A-D])\.",
        r"^([A-D])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()

    return None


def gpqa_eval_fn(expected: str, actual: Any) -> float:
    """Return 1.0 if the extracted answer matches *expected*, else 0.0."""
    actual_str = extract_text_content(actual)
    extracted = extract_answer_letter(actual_str)

    if extracted is None:
        logger.warning(
            "Could not extract answer letter from response (expected=%s): %s",
            expected,
            actual_str[:200],
        )
        return 0.0

    correct = extracted == expected.upper()
    if not correct:
        logger.debug(
            "Wrong answer: expected=%s, extracted=%s, response=%s",
            expected,
            extracted,
            actual_str[:200],
        )

    return 1.0 if correct else 0.0


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = (
    "You are an expert scientist answering a multiple-choice question. "
    "Think through the problem step by step, then state your final "
    "answer as a single letter (A, B, C, or D) on the last line, "
    "formatted as: Answer: X"
)

SELF_REFLECT_SYSTEM_PROMPT = COT_SYSTEM_PROMPT

SELF_REFLECT_CRITIQUE_PROMPT = (
    "Review your previous answer. Re-examine your reasoning for errors. "
    "If your answer is correct, reply: CONFIRMED: X (the letter). "
    "If wrong, explain the error and provide a corrected answer as: Answer: X"
)


# ---------------------------------------------------------------------------
# Agent factories — raw mode (direct API calls)
# ---------------------------------------------------------------------------


def _gpqa_agent_fn_raw(models: Dict[str, Any], agent_type: str = "direct", max_iterations: int = 3):
    """Factory: raw mode agent_fn for GPQA.

    models: {"agent": "<model display name or ARN>"}
    Returns a callable: (input_data) -> response_string
    """
    model_spec = models["agent"]

    def run(input_data: dict) -> str:
        llm = make_llm(model_spec) if isinstance(model_spec, str) else model_spec
        prompt = input_data["input"]

        if agent_type == "direct":
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return extract_text_content(content)

        elif agent_type == "cot":
            messages = [
                SystemMessage(content=COT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return extract_text_content(content)

        else:  # self_reflect
            messages = [
                SystemMessage(content=SELF_REFLECT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            messages.append(response)

            for _ in range(max_iterations):
                messages.append(HumanMessage(content=SELF_REFLECT_CRITIQUE_PROMPT))
                response = llm.invoke(messages)
                messages.append(response)
                content = extract_text_content(
                    response.content if hasattr(response, "content") else str(response)
                )
                if "CONFIRMED" in content.upper():
                    return content

            content = response.content if hasattr(response, "content") else str(response)
            return extract_text_content(content)

    return run


# ---------------------------------------------------------------------------
# Agent factories — LangGraph mode
# ---------------------------------------------------------------------------


def _gpqa_agent_fn_langgraph(models: Dict[str, Any], agent_type: str = "direct", max_iterations: int = 3):
    """Factory: LangGraph mode agent_fn for GPQA.

    models: {"agent": "<model display name or ARN>"}
    Returns a callable: (input_data) -> response_string
    """
    model_spec = models["agent"]

    if agent_type in ("direct", "cot"):
        system_msg = None
        if agent_type == "cot":
            system_msg = SystemMessage(content=COT_SYSTEM_PROMPT)

        def run(input_data: dict) -> str:
            llm = make_llm(model_spec) if isinstance(model_spec, str) else model_spec

            def call_model(state: MessagesState):
                response = llm.invoke(state["messages"])
                return {"messages": [response]}

            graph = StateGraph(MessagesState)
            graph.add_node("agent", call_model)
            graph.set_entry_point("agent")
            graph.add_edge("agent", END)
            compiled = graph.compile()

            messages = []
            if system_msg:
                messages.append(system_msg)
            messages.append(HumanMessage(content=input_data["input"]))
            result = compiled.invoke({"messages": messages})
            last = result["messages"][-1]
            content = last.content if hasattr(last, "content") else str(last)
            return extract_text_content(content)

        return run

    else:  # self_reflect
        from typing import List, TypedDict

        class ReflectState(TypedDict):
            messages: List[Any]
            final: str
            iteration: int

        def run(input_data: dict) -> str:
            llm = make_llm(model_spec) if isinstance(model_spec, str) else model_spec

            def init_node(state: ReflectState) -> ReflectState:
                return {
                    "messages": [
                        SystemMessage(content=SELF_REFLECT_SYSTEM_PROMPT),
                        HumanMessage(content=input_data["input"]),
                    ],
                    "final": "",
                    "iteration": 0,
                }

            def answer_node(state: ReflectState) -> ReflectState:
                messages = list(state["messages"])
                response = llm.invoke(messages)
                messages.append(response)
                return {"messages": messages, "final": "", "iteration": state["iteration"]}

            def critique_node(state: ReflectState) -> ReflectState:
                messages = list(state["messages"])
                messages.append(HumanMessage(content=SELF_REFLECT_CRITIQUE_PROMPT))
                response = llm.invoke(messages)
                messages.append(response)
                iteration = state["iteration"] + 1
                content = extract_text_content(
                    response.content if hasattr(response, "content") else str(response)
                )
                final = content if "CONFIRMED" in content.upper() else ""
                return {"messages": messages, "final": final, "iteration": iteration}

            def route(state: ReflectState) -> str:
                if state["final"]:
                    return "finalize"
                if state["iteration"] >= max_iterations:
                    return "finalize"
                return "answer_node"

            def finalize(state: ReflectState) -> ReflectState:
                messages = list(state["messages"])
                final = state.get("final", "")
                if not final and messages:
                    last = messages[-1]
                    content = last.content if hasattr(last, "content") else str(last)
                    final = extract_text_content(content)
                return {"messages": messages, "final": final, "iteration": state["iteration"]}

            graph = StateGraph(ReflectState)
            graph.add_node("init", init_node)
            graph.add_node("answer_node", answer_node)
            graph.add_node("critique_node", critique_node)
            graph.add_node("finalize", finalize)

            graph.set_entry_point("init")
            graph.add_edge("init", "answer_node")
            graph.add_edge("answer_node", "critique_node")
            graph.add_conditional_edges(
                "critique_node", route,
                {"finalize": "finalize", "answer_node": "answer_node"},
            )
            graph.add_edge("finalize", END)
            compiled = graph.compile()

            result = compiled.invoke({
                "messages": [HumanMessage(content=input_data["input"])],
                "final": "",
                "iteration": 0,
            })
            return result.get("final", "")

        return run


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results, title="GPQA Model Performance", save_path=None):
    plt.figure(figsize=(10, 6))
    accuracies = [r.accuracy for r in results]
    latencies = [r.latency_seconds for r in results]
    names = [r.model_name for r in results]

    plt.scatter(latencies, accuracies, s=100, zorder=5)
    for name, lat, acc in zip(names, latencies, accuracies):
        plt.annotate(
            display_name(name),
            (lat, acc),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

AGENT_TYPES = ("direct", "cot", "self_reflect")


def main():
    parser = argparse.ArgumentParser(
        description="GPQA Diamond Benchmark with AgentOpt (Bedrock)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.GPQA.eval --limit 10\n"
            "  python -m benchmarks.GPQA.eval --all-models --limit 40 --no-cache --parallel\n"
            "  python -m benchmarks.GPQA.eval --agent cot --limit 20\n"
        ),
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--agent",
        choices=AGENT_TYPES,
        default="direct",
        help="Agent type: 'direct', 'cot', or 'self_reflect'",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max self-reflection iterations (only for self_reflect, default: 3)",
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
    print(f"GPQA Diamond Benchmark  [{mode_label}]")
    print("=" * 60)

    # Dataset
    print(f"\n[1] Loading GPQA Diamond dataset ...")
    dataset = load_gpqa_dataset(limit=args.limit)
    print(f"    {len(dataset)} questions loaded")

    domains: dict[str, int] = {}
    for input_data, _ in dataset:
        d = input_data.get("domain", "Unknown")
        domains[d] = domains.get(d, 0) + 1
    for domain, count in sorted(domains.items()):
        print(f"    - {domain}: {count}")

    # Agent factory
    agent_label = args.agent
    print(f"\n[2] Setting up agent: {agent_label} ({mode_label})")
    print(f"    Models: {models}")

    if args.mode == "raw":
        def agent_fn(m):
            return _gpqa_agent_fn_raw(m, agent_type=args.agent, max_iterations=args.max_iterations)
    else:
        def agent_fn(m):
            return _gpqa_agent_fn_langgraph(m, agent_type=args.agent, max_iterations=args.max_iterations)

    # Model selection
    print(f"\n[3] Running model selection ({args.selector}) ...")
    selector_kwargs = build_selector_kwargs(args)

    from agentopt import LLMTracker

    SelectorCls = SELECTORS[args.selector]
    selector = SelectorCls(
        agent=agent_fn,
        models={"agent": models},
        eval_fn=gpqa_eval_fn,
        dataset=dataset,
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
    print(f"GPQA Diamond Results ({agent_label}, {mode_label})")
    print(f"{'=' * 60}")
    print(f"Questions: {len(dataset)} | Selector: {args.selector} | Time: {elapsed:.1f}s")
    print()

    results.print_summary()

    if args.csv:
        results.to_csv(args.csv)
        print(f"Results saved to: {args.csv}")

    if args.plot:
        plot_results(results, f"GPQA Diamond — {agent_label}", args.plot)


if __name__ == "__main__":
    main()
