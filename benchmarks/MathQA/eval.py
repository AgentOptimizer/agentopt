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
    convert_to_messages,
)
from langgraph.graph import END, StateGraph

from benchmarks.MathQA.tools import TOOLS as CALC_TOOLS
from benchmarks.runtime import (
    ToolLimits, ToolRunState, parse_math_answer, response_stop_reason, run_with_tools,
)

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


def _portable_message_history(messages: List[Any]) -> List[Any]:
    """Remove provider-private reasoning blocks before a cross-model handoff.

    Bedrock reasoning blocks (including ``redacted_thinking`` and
    ``reasoning_content``) are only valid when replayed to the model/provider
    that created them.  MathQA alternates answer and critic models, so passing
    those opaque blocks through unchanged can make the next Converse request
    fail validation.  Visible text, tool calls, and tool results remain intact.
    """

    def portable_content(content: Any) -> Any:
        if not isinstance(content, list):
            return content
        kept: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                kept.append(block)
                continue
            block_type = str(block.get("type", "")).lower()
            block_keys = {str(key).lower() for key in block}
            if (
                "reasoning" in block_type
                or "thinking" in block_type
                or "reasoningcontent" in block_keys
                or "redactedcontent" in block_keys
            ):
                continue
            kept.append(block)
        return kept

    portable: list[Any] = []
    for message in messages:
        content = (
            message.get("content", "")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        cleaned = portable_content(content)
        if isinstance(message, dict):
            copied = dict(message)
            copied["content"] = cleaned
            portable.append(copied)
            continue
        if hasattr(message, "model_copy"):
            updates: dict[str, Any] = {"content": cleaned}
            if getattr(message, "type", None) == "ai":
                # Parsed tool calls live in ``tool_calls``; provider-specific
                # raw reasoning metadata is not safe to replay cross-model.
                updates["additional_kwargs"] = {}
            portable.append(message.model_copy(update=updates))
            continue
        portable.append(message)
    return portable


def _is_system_message(message: Any) -> bool:
    if isinstance(message, dict):
        return message.get("role") == "system"
    if hasattr(message, "type"):
        return message.type == "system"
    return False


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        # Structured results must identify the Answerer's output explicitly.
        # An arbitrary last message can belong to the critic or a tool.
        return extract_text_content(result.get("answer_text", result.get("final", "")))
    return extract_text_content(result)


def eval_fn(expected: str, actual: Any) -> bool:
    """Score only a valid explicit choice from the final Answerer response."""
    if isinstance(actual, dict) and actual.get("format_valid") is False:
        return False
    reason = actual.get("stop_reason") if isinstance(actual, dict) else None
    parsed = parse_math_answer(_extract_text(actual), reason)
    return parsed["format_valid"] and parsed["answer_option"] == expected.strip().lower()


# ---------------------------------------------------------------------------
# Tool call helper
# ---------------------------------------------------------------------------


def _run_answer_with_tools(
    llm, messages, use_tools=True, max_tool_rounds=8, *, tool_limits=None, tool_state=None,
):
    return run_with_tools(
        llm, messages, CALC_TOOLS, use_tools=use_tools,
        limits=tool_limits or ToolLimits(max_tool_rounds=max_tool_rounds),
        state=tool_state, prepare_messages=_portable_message_history,
    )


def _answer_snapshot(response, iteration):
    text = extract_text_content(getattr(response, "content", ""))
    reason = "tool_use" if getattr(response, "tool_calls", None) else response_stop_reason(response)
    return {
        **parse_math_answer(text, reason),
        "final": text.strip(),
        "answer_source": "answerer",
        "answer_iteration": iteration,
    }


def _mathqa_agent_fn_raw(
    models: Dict[str, Any], max_iterations=3, max_tool_rounds=8, *, tool_limits=None,
):
    """Run reflection with a dedicated final Answerer record, separate from reviews."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    answer_spec, critic_spec = models["answer"], models["critic"]
    limits = tool_limits or ToolLimits(max_tool_rounds=max_tool_rounds)

    def run(input_data: dict) -> dict:
        answer_llm = make_llm(answer_spec) if isinstance(answer_spec, str) else answer_spec
        critic_llm = make_llm(critic_spec) if isinstance(critic_spec, str) else critic_spec
        use_tools = supports_tool_calling(answer_spec) if isinstance(answer_spec, str) else True
        system = ANSWER_SYSTEM_PROMPT if use_tools else ANSWER_SYSTEM_PROMPT_NO_TOOLS
        messages = _normalize_messages(list(input_data.get("messages", [])))
        if not messages or not _is_system_message(messages[0]):
            messages = [SystemMessage(content=system)] + messages
        tool_state = ToolRunState()
        answer_rounds = []
        for iteration in range(1, max_iterations + 1):
            messages, response = _run_answer_with_tools(
                answer_llm, messages, use_tools, tool_limits=limits, tool_state=tool_state,
            )
            snapshot = _answer_snapshot(response, iteration)
            answer_rounds.append(dict(snapshot))
            critic_messages = [SystemMessage(content=CRITIC_SYSTEM_PROMPT)] + messages[1:] + [
                HumanMessage(content=(
                    "Review the previous answer. Verify it by solving the "
                    "problem yourself. Reply CORRECT if the answer is right, "
                    "or INCORRECT: <feedback> if it is wrong."
                ))
            ]
            review = critic_llm.invoke(_portable_message_history(critic_messages))
            messages.append(review)
            content = extract_text_content(getattr(review, "content", ""))
            if "CORRECT" in content.upper() and "INCORRECT" not in content.upper():
                break
            if iteration < max_iterations:
                messages.append(HumanMessage(content=(
                    "Revise your answer using the critic feedback above. Return a complete corrected answer."
                )))
        return {"messages": messages, **snapshot, "iteration": iteration,
                "answer_rounds": answer_rounds, **tool_state.metadata()}

    return run


class ReflectionState(TypedDict, total=False):
    messages: List[Any]
    final: str
    iteration: int
    answer_text: str
    answer_option: str | None
    answer_iteration: int
    answer_source: str
    format_valid: bool
    format_failure: str | None
    stop_reason: str | None
    empty_answer: bool
    output_truncated: bool
    answer_rounds: list
    tool_call_count: int
    tool_events: list


def _mathqa_agent_fn_langgraph(
    models: Dict[str, Any], max_iterations=3, max_tool_rounds=8, *, tool_limits=None,
):
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    answer_spec, critic_spec = models["answer"], models["critic"]
    limits = tool_limits or ToolLimits(max_tool_rounds=max_tool_rounds)

    def run(input_data: dict) -> dict:
        answer_llm = make_llm(answer_spec) if isinstance(answer_spec, str) else answer_spec
        critic_llm = make_llm(critic_spec) if isinstance(critic_spec, str) else critic_spec
        use_tools = supports_tool_calling(answer_spec) if isinstance(answer_spec, str) else True
        system = ANSWER_SYSTEM_PROMPT if use_tools else ANSWER_SYSTEM_PROMPT_NO_TOOLS
        # State is local to this question, including when a factory is used concurrently.
        tool_state = ToolRunState()

        def add_system(state):
            messages = _normalize_messages(list(state.get("messages", [])))
            if not messages or not _is_system_message(messages[0]):
                messages = [SystemMessage(content=system)] + messages
            return {"messages": messages, "iteration": 0, "answer_rounds": []}

        def call_answer_agent(state):
            messages = list(state["messages"])
            if messages and getattr(messages[-1], "type", None) == "ai":
                messages.append(HumanMessage(content=(
                    "Revise your answer using the critic feedback above. Return a complete corrected answer."
                )))
            messages, response = _run_answer_with_tools(
                answer_llm, messages, use_tools, tool_limits=limits, tool_state=tool_state,
            )
            snapshot = _answer_snapshot(response, state["iteration"] + 1)
            return {**state, **snapshot, "messages": messages,
                    "answer_rounds": state["answer_rounds"] + [dict(snapshot)], **tool_state.metadata()}

        def call_critic_agent(state):
            messages = list(state["messages"])
            critic_messages = [SystemMessage(content=CRITIC_SYSTEM_PROMPT)] + messages[1:] + [
                HumanMessage(content="Review the previous answer. Reply CORRECT or INCORRECT: <feedback>.")
            ]
            response = critic_llm.invoke(_portable_message_history(critic_messages))
            return {**state, "messages": messages + [response], "iteration": state["iteration"] + 1}

        def route_after_critic(state):
            content = extract_text_content(getattr(state["messages"][-1], "content", ""))
            approved = "CORRECT" in content.upper() and "INCORRECT" not in content.upper()
            return END if approved or state["iteration"] >= max_iterations else "call_answer_agent"

        graph = StateGraph(ReflectionState)
        graph.add_node("add_system", add_system)
        graph.add_node("call_answer_agent", call_answer_agent)
        graph.add_node("call_critic_agent", call_critic_agent)
        graph.set_entry_point("add_system")
        graph.add_edge("add_system", "call_answer_agent")
        graph.add_edge("call_answer_agent", "call_critic_agent")
        graph.add_conditional_edges("call_critic_agent", route_after_critic,
                                    {END: END, "call_answer_agent": "call_answer_agent"})
        # Three nodes per reflection at most, plus graph start/termination steps.
        return graph.compile().invoke(input_data, config={"recursion_limit": 3 * max_iterations + 5})

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
    from benchmarks.runtime import add_tool_limit_args, tool_limits_from_args
    add_tool_limit_args(parser)
    args = parser.parse_args()
    tool_limits = tool_limits_from_args(args)

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
            return _mathqa_agent_fn_raw(m, max_iterations=args.max_iterations, tool_limits=tool_limits)
    else:
        def agent_fn(m):
            return _mathqa_agent_fn_langgraph(m, max_iterations=args.max_iterations, tool_limits=tool_limits)

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
