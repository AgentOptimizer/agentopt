"""
HotpotQA (Distractor) Benchmark — Bedrock edition.

2-tuple benchmark: planner + solver with search tools.
Multi-hop QA with wikipedia_search, web_search, and lookup_keyword tools.

Supports raw (direct API) and langgraph modes.
Supports single (1-tuple) and multi (2-tuple planner+solver) pipelines.

Dataset: HotpotQA distractor JSON (local file).

Usage:
    python -m benchmarks.HotpotQA.eval --dataset path/to/hotpotqa.json --limit 50
    python -m benchmarks.HotpotQA.eval --dataset data.json --all-models --no-cache --parallel
    python -m benchmarks.HotpotQA.eval --dataset data.json --mode raw --limit 30
    python -m benchmarks.HotpotQA.eval --dataset data.json --pipeline single --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGSMITH_TRACING", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from benchmarks.common import (
    add_common_cli_args,
    build_selector_kwargs,
    display_name,
    extract_text_content,
    get_selectors,
    make_llm,
    resolve_models,
    supports_tool_calling,
)
from benchmarks.HotpotQA.tools import TOOLS as SEARCH_TOOLS

logger = logging.getLogger(__name__)

SELECTORS = get_selectors()

Dataset = List[Tuple[Dict[str, Any], str]]

INSTRUCTIONS = (
    "Answer the question using the provided context.\n"
    "You have access to search tools: wikipedia_search, web_search, and lookup_keyword.\n"
    "Use them when you need additional information or to locate specific facts in the context.\n"
    "Return ONLY the final answer string (no extra words, no explanation)."
)

INSTRUCTIONS_NO_TOOLS = (
    "Answer the question using the provided context.\n"
    "Return ONLY the final answer string (no extra words, no explanation)."
)

PLANNER_INSTRUCTIONS = (
    "You are a planner in a planner/solver QA loop.\n"
    "Given context, question, and prior solver output, propose ONE concrete next step.\n"
    "If enough evidence is available, return terminate() and include a concise final answer.\n"
    "Format:\n"
    "- Ongoing: NEXT: <single concrete step>\n"
    "- Finish: terminate()\\nFINAL: <answer>"
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _format_context(context: Any) -> str:
    blocks: List[str] = []
    if not isinstance(context, list):
        return str(context)
    for entry in context:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], list)
        ):
            title = entry[0].strip()
            sentences = [
                str(sentence).strip() for sentence in entry[1] if str(sentence).strip()
            ]
            text = " ".join(sentences)
            blocks.append(f"[{title}] {text}".strip() if title else text)
        else:
            blocks.append(str(entry).strip())
    return "\n".join(block for block in blocks if block)


def load_hotpotqa_distractor(
    dataset_path: str,
    *,
    limit: int | None = None,
    seed: int = 0,
    context_max_chars: int | None = None,
) -> Dataset:
    """Load and shuffle HotpotQA distractor data."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = _read_json(path)
    raw_items = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    if not isinstance(raw_items, list):
        raise ValueError(f"Unexpected dataset format in {path}")

    rng = random.Random(seed)
    items = list(raw_items)
    rng.shuffle(items)
    if limit is not None:
        items = items[: max(0, int(limit))]

    dataset: Dataset = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        context = _format_context(item.get("context"))
        if context_max_chars is not None and context_max_chars > 0:
            context = context[:context_max_chars]
        if not question or not answer or not context:
            continue
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        dataset.append(({"input": prompt}, answer))

    if not dataset:
        raise ValueError("No usable samples loaded from dataset")
    return dataset


def split_dataset(
    dataset: Dataset,
    *,
    selection_ratio: float = 0.2,
    selection_size: int | None = None,
) -> Tuple[Dataset, Dataset]:
    """Split samples into model-selection and holdout subsets."""
    total = len(dataset)
    if total < 2:
        raise ValueError("Need at least 2 samples.")

    if selection_size is not None and selection_size > 0:
        selection_count = selection_size
    else:
        selection_count = int(round(total * selection_ratio))

    selection_count = max(1, min(total - 1, selection_count))
    return dataset[:selection_count], dataset[selection_count:]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def hotpot_em(gold: str, pred: str) -> float:
    return 1.0 if _normalize_answer(gold) == _normalize_answer(pred) else 0.0


def hotpot_f1(gold: str, pred: str) -> float:
    gold_toks = _normalize_answer(gold).split()
    pred_toks = _normalize_answer(pred).split()
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0

    overlap: Dict[str, int] = {}
    for token in gold_toks:
        overlap[token] = overlap.get(token, 0) + 1

    num_same = 0
    for token in pred_toks:
        if overlap.get(token, 0) > 0:
            num_same += 1
            overlap[token] -= 1
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("output", "final_output", "text", "answer", "final"):
            if key in result:
                return str(result[key])
        if "messages" in result and result["messages"]:
            last = result["messages"][-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
            if hasattr(last, "content"):
                return str(last.content)
    if hasattr(result, "final_output"):
        return str(result.final_output)
    return str(result)


# ---------------------------------------------------------------------------
# Planner/solver control loop helpers
# ---------------------------------------------------------------------------


def _build_solver_input(prompt: str, planner_notes: str) -> str:
    return (
        f"{prompt}\n\n"
        f"Planner notes:\n{planner_notes}\n\n"
        "Now provide only the final answer."
    )


def _planner_requests_terminate(planner_text: str) -> bool:
    return bool(
        re.search(r"\bterminate\s*\(\s*\)", planner_text or "", flags=re.IGNORECASE)
    )


def _extract_planner_final(planner_text: str) -> str:
    text = (planner_text or "").strip()
    if not text:
        return ""
    final_match = re.search(
        r"\bfinal(?:\s+answer)?\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        return final_match.group(1).strip()
    if _planner_requests_terminate(text):
        tail = re.split(r"\bterminate\s*\(\s*\)", text, flags=re.IGNORECASE, maxsplit=1)
        if len(tail) == 2:
            return tail[1].strip(" \n:-")
    return ""


def _build_planner_input(
    prompt: str,
    previous_solver_output: str,
    reflection_idx: int,
    max_reflections: int,
) -> str:
    if reflection_idx == 0:
        return (
            f"{prompt}\n\n"
            "No prior solver output yet. Propose the first concrete next step."
        )
    return (
        f"{prompt}\n\n"
        f"Previous solver output:\n{previous_solver_output}\n\n"
        f"Reflection step {reflection_idx + 1} of {max_reflections}. "
        "Propose the next concrete step, or return terminate() with FINAL: <answer>."
    )


def _invoke_with_tools(llm, new_messages, prior_messages=None, use_tools=True):
    """Multi-turn tool loop, accumulating history across calls."""
    messages = list(prior_messages or []) + list(new_messages)

    if use_tools:
        tools_by_name = {t.name: t for t in SEARCH_TOOLS}
        model_with_tools = llm.bind_tools(SEARCH_TOOLS)
        response = model_with_tools.invoke(messages)
        messages.append(response)
        while getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                tool_fn = tools_by_name.get(tc["name"])
                if tool_fn:
                    tool_result = tool_fn.invoke(tc["args"])
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                    )
            response = model_with_tools.invoke(messages)
            messages.append(response)
    else:
        response = llm.invoke(messages)
        messages.append(response)

    return str(getattr(response, "content", response)), messages


def _run_reflection_loop(
    prompt: str,
    planner_call,
    solver_call,
    max_reflections: int,
    stop_policy: str,
) -> str:
    max_reflections = max(1, int(max_reflections))
    last_solver_output = ""
    solver_messages: list = []
    for reflection_idx in range(max_reflections):
        planner_input = _build_planner_input(
            prompt, last_solver_output, reflection_idx, max_reflections
        )
        planner_notes = str(planner_call(planner_input)).strip()

        if _planner_requests_terminate(planner_notes):
            planner_final = _extract_planner_final(planner_notes)
            if planner_final:
                return planner_final
            final_text, solver_messages = solver_call(
                _build_solver_input(prompt, planner_notes), solver_messages
            )
            final_text = final_text.strip()
            return final_text if final_text else planner_notes

        answer_text, solver_messages = solver_call(
            _build_solver_input(prompt, planner_notes), solver_messages
        )
        answer_text = answer_text.strip()
        if (
            stop_policy == "converged"
            and answer_text
            and _normalize_answer(answer_text) == _normalize_answer(last_solver_output)
        ):
            return answer_text
        last_solver_output = answer_text
    return last_solver_output


# ---------------------------------------------------------------------------
# Raw mode agent_fn factory
# ---------------------------------------------------------------------------


def _hotpotqa_agent_fn_raw(
    models: Dict[str, Any],
    pipeline: str = "multi",
    max_reflections: int = 1,
    reflection_stop: str = "converged",
):
    """Factory: raw mode agent_fn for HotpotQA.

    Single pipeline: models = {"agent": "<model>"}
    Multi pipeline:  models = {"planner": "<model>", "solver": "<model>"}
    """
    if pipeline == "single":
        model_spec = models["agent"]
        use_tools = supports_tool_calling(model_spec) if isinstance(model_spec, str) else True
        instructions = INSTRUCTIONS if use_tools else INSTRUCTIONS_NO_TOOLS

        def run(input_data: Dict[str, Any]) -> str:
            llm = make_llm(model_spec) if isinstance(model_spec, str) else model_spec
            answer, _ = _invoke_with_tools(
                llm,
                [
                    SystemMessage(content=instructions),
                    HumanMessage(content=input_data["input"][:12000]),
                ],
                use_tools=use_tools,
            )
            return answer

        return run

    # Multi pipeline
    planner_spec = models["planner"]
    solver_spec = models["solver"]
    use_tools = supports_tool_calling(solver_spec) if isinstance(solver_spec, str) else True
    instructions = INSTRUCTIONS if use_tools else INSTRUCTIONS_NO_TOOLS

    def run(input_data: Dict[str, Any]) -> str:
        planner_llm = make_llm(planner_spec) if isinstance(planner_spec, str) else planner_spec
        solver_llm = make_llm(solver_spec) if isinstance(solver_spec, str) else solver_spec

        def planner_call(prompt: str) -> str:
            response = planner_llm.invoke(
                [
                    SystemMessage(content=PLANNER_INSTRUCTIONS),
                    HumanMessage(content=prompt[:12000]),
                ]
            )
            return str(getattr(response, "content", response))

        def solver_call(prompt, prior_messages=None):
            return _invoke_with_tools(
                solver_llm,
                [
                    SystemMessage(content=instructions),
                    HumanMessage(content=prompt[:12000]),
                ],
                prior_messages,
                use_tools=use_tools,
            )

        return _run_reflection_loop(
            prompt=input_data["input"],
            planner_call=planner_call,
            solver_call=solver_call,
            max_reflections=max_reflections,
            stop_policy=reflection_stop,
        )

    return run


# ---------------------------------------------------------------------------
# LangGraph mode agent_fn factory
# ---------------------------------------------------------------------------


def _hotpotqa_agent_fn_langgraph(
    models: Dict[str, Any],
    pipeline: str = "multi",
    max_reflections: int = 1,
    reflection_stop: str = "converged",
):
    """Factory: LangGraph mode agent_fn for HotpotQA."""
    from langgraph.graph import END, StateGraph

    if pipeline == "single":
        model_spec = models["agent"]
        use_tools = supports_tool_calling(model_spec) if isinstance(model_spec, str) else True
        instructions = INSTRUCTIONS if use_tools else INSTRUCTIONS_NO_TOOLS

        def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
            llm = make_llm(model_spec) if isinstance(model_spec, str) else model_spec

            def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
                prompt = state["input"][:12000]
                answer, _ = _invoke_with_tools(
                    llm,
                    [SystemMessage(content=instructions), HumanMessage(content=prompt)],
                    use_tools=use_tools,
                )
                return {"input": state["input"], "final": answer}

            graph = StateGraph(dict)
            graph.add_node("answer", answer_node)
            graph.set_entry_point("answer")
            graph.add_edge("answer", END)
            compiled = graph.compile()

            return compiled.invoke({"input": input_data["input"]})

        return run

    # Multi pipeline — same reflection loop but with LangGraph-created LLMs
    planner_spec = models["planner"]
    solver_spec = models["solver"]
    use_tools = supports_tool_calling(solver_spec) if isinstance(solver_spec, str) else True
    instructions = INSTRUCTIONS if use_tools else INSTRUCTIONS_NO_TOOLS

    def run(input_data: Dict[str, Any]) -> str:
        planner_llm = make_llm(planner_spec) if isinstance(planner_spec, str) else planner_spec
        solver_llm = make_llm(solver_spec) if isinstance(solver_spec, str) else solver_spec

        def planner_call(prompt: str) -> str:
            response = planner_llm.invoke(
                [
                    SystemMessage(content=PLANNER_INSTRUCTIONS),
                    HumanMessage(content=prompt[:12000]),
                ]
            )
            return str(getattr(response, "content", response))

        def solver_call(prompt, prior_messages=None):
            return _invoke_with_tools(
                solver_llm,
                [
                    SystemMessage(content=instructions),
                    HumanMessage(content=prompt[:12000]),
                ],
                prior_messages,
                use_tools=use_tools,
            )

        return _run_reflection_loop(
            prompt=input_data["input"],
            planner_call=planner_call,
            solver_call=solver_call,
            max_reflections=max_reflections,
            stop_policy=reflection_stop,
        )

    return run


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results, title="HotpotQA Model Performance", save_path=None):
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
    plt.ylabel("Score (F1/EM)")
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


def main():
    parser = argparse.ArgumentParser(
        description="HotpotQA Distractor Benchmark with AgentOpt (Bedrock)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.HotpotQA.eval --dataset data.json --limit 50\n"
            "  python -m benchmarks.HotpotQA.eval --dataset data.json --all-models --no-cache --parallel\n"
            "  python -m benchmarks.HotpotQA.eval --dataset data.json --mode raw --pipeline single\n"
        ),
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--dataset", required=True,
        help="Path to HotpotQA distractor JSON file",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--context-max-chars", type=int, default=12000,
        help="Max context chars per sample (default: 12000)",
    )
    parser.add_argument(
        "--metric", choices=["f1", "em"], default="f1",
        help="Evaluation metric (default: f1)",
    )
    parser.add_argument(
        "--pipeline", choices=["single", "multi"], default="multi",
        help="single = one-step; multi = planner + solver (default: multi)",
    )
    parser.add_argument(
        "--max-reflections", type=int, default=1,
        help="Max planner->solver reflection iterations (default: 1)",
    )
    parser.add_argument(
        "--reflection-stop", choices=["none", "converged"], default="converged",
        help="Early-stop policy for reflections (default: converged)",
    )
    parser.add_argument(
        "--selection-ratio", type=float, default=0.2,
        help="Fraction for model selection (default: 0.2)",
    )
    parser.add_argument(
        "--selection-size", type=int, default=0,
        help="Fixed selection set size (overrides ratio if > 0)",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    models = resolve_models(args)
    mode_label = "Raw API" if args.mode == "raw" else "LangGraph"
    metric_fn = hotpot_f1 if args.metric == "f1" else hotpot_em

    print("=" * 60)
    print(f"HotpotQA Distractor Benchmark  [{mode_label}]")
    print("=" * 60)

    # Dataset
    print("\n[1] Loading HotpotQA distractor dataset...")
    all_samples = load_hotpotqa_distractor(
        args.dataset,
        limit=args.limit,
        seed=args.seed,
        context_max_chars=(
            None if args.context_max_chars == 0 else args.context_max_chars
        ),
    )
    selection_set, test_set = split_dataset(
        all_samples,
        selection_ratio=args.selection_ratio,
        selection_size=args.selection_size if args.selection_size > 0 else None,
    )
    print(f"    Total: {len(all_samples)}, Selection: {len(selection_set)}, Test: {len(test_set)}")

    # Eval function
    def eval_fn(expected: str, actual: Any) -> float:
        return metric_fn(expected, _extract_text(actual))

    # Agent factory
    print(f"\n[2] Setting up agents ({mode_label}, pipeline={args.pipeline})...")
    print(f"    Models: {models}")

    if args.pipeline == "single":
        models_config = {"agent": models}
    else:
        models_config = {"planner": models, "solver": models}
        print(f"    Search space: {len(models)}x{len(models)} = {len(models)**2} combos")

    if args.mode == "raw":
        def agent_fn(m):
            return _hotpotqa_agent_fn_raw(
                m, pipeline=args.pipeline,
                max_reflections=args.max_reflections,
                reflection_stop=args.reflection_stop,
            )
    else:
        def agent_fn(m):
            return _hotpotqa_agent_fn_langgraph(
                m, pipeline=args.pipeline,
                max_reflections=args.max_reflections,
                reflection_stop=args.reflection_stop,
            )

    # Model selection
    print(f"\n[3] Running model selection ({args.selector})...")
    selector_kwargs = build_selector_kwargs(args)

    from agentopt import LLMTracker

    SelectorCls = SELECTORS[args.selector]
    selector = SelectorCls(
        agent_fn=agent_fn,
        models=models_config,
        eval_fn=eval_fn,
        dataset=selection_set,
        # model_price.json handles pricing for OpenRouter model names
        **selector_kwargs,
    )

    if args.no_cache:
        selector._tracker.stop()
        tracker = LLMTracker(cache=False)
        selector._tracker = tracker
        tracker.start()

    start = time.time()
    selection_results = selector.select_best(parallel=args.parallel)
    elapsed = time.time() - start

    # Results
    print(f"\n{'=' * 60}")
    print(f"HotpotQA Results ({mode_label}, {args.metric})")
    print(f"{'=' * 60}")
    print(f"Selection: {len(selection_set)} | Selector: {args.selector} | Time: {elapsed:.1f}s")
    print()

    selection_results.print_summary()

    best = selection_results.get_best()
    if best:
        print(f"\n[best] {best.model_name} ({args.metric}={best.accuracy:.3f})")

    # Holdout evaluation
    print(f"\n[4] Holdout evaluation ({len(test_set)} samples)...")
    if best:
        holdout_scores, holdout_latencies, _ = selector._evaluate_sequential(
            test_set, label="holdout"
        )
        holdout_score, _ = selector._compute_stats(holdout_scores)
        holdout_latency = (
            sum(holdout_latencies) / len(holdout_latencies)
            if holdout_latencies
            else 0.0
        )
        print(
            f"[holdout] {args.metric}={holdout_score:.3f} "
            f"avg_latency={holdout_latency:.2f}s over {len(test_set)} samples"
        )

    if args.csv:
        selection_results.to_csv(args.csv)
        print(f"Results saved to: {args.csv}")

    if args.plot:
        plot_results(
            selection_results,
            f"HotpotQA — {mode_label} ({args.metric})",
            args.plot,
        )


if __name__ == "__main__":
    main()
