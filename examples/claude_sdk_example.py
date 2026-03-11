"""Claude Agent SDK examples with AgentOpt model selection."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentopt import ModelProxy
from agentopt.model_selection import (
    BruteForceModelSelector,
    RandomSearchModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
    BayesianOptimizationModelSelector,
)

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random_search": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "bayesian_optimization": BayesianOptimizationModelSelector,
}


def load_dataset(dataset_dir, filename):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for Claude Agent SDK."""
    dataset_path = Path(dataset_dir)
    jsonl_file = dataset_path / filename

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks


def eval_fn(expected, actual):
    return expected.lower() in str(actual).lower()


async def _query_async(prompt, options):
    """Run a single Claude Agent SDK query and return the result text."""
    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, "result"):
            result_text = message.result
    return result_text


def single_agent_example():
    """Single agent wrapped with ModelProxy for model selection."""
    proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))
    print("  [setup] proxy created (initial model: haiku)")

    def invoke_fn(input_data):
        return asyncio.run(_query_async(input_data["input"], proxy))

    return proxy, invoke_fn, None


def multi_agent_example():
    """Multi-agent chain with a shared LLM.

    Solver answers the question, then Reviewer verifies the answer.
    Both use the same ModelProxy so they are optimized together.
    """
    proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))
    print("  [setup] proxy created (initial model: haiku)")
    print("  [setup] pipeline: solver -> reviewer (shared proxy)")

    def invoke_fn(input_data):
        question = input_data["input"]
        answer = asyncio.run(
            _query_async(f"Solve this math problem: {question}", proxy)
        )
        verified = asyncio.run(
            _query_async(
                f"Verify this answer to '{question}': {answer}. "
                f"Reply with just the final number.",
                proxy,
            )
        )
        return verified

    return proxy, invoke_fn, None


def multi_agent_multi_llm_example():
    """Multi-agent chain with separate LLMs per agent.

    Solver and Reviewer each have their own ModelProxy so they can be
    optimized independently by BruteForceModelSelector.
    """
    solver_proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))
    print("  [setup] solver_proxy created (initial model: haiku)")
    reviewer_proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))
    print("  [setup] reviewer_proxy created (initial model: haiku)")
    print("  [setup] pipeline: solver (proxy1) -> reviewer (proxy2)")

    def invoke_fn(input_data):
        question = input_data["input"]
        answer = asyncio.run(
            _query_async(f"Solve this math problem: {question}", solver_proxy)
        )
        verified = asyncio.run(
            _query_async(
                f"Verify this answer to '{question}': {answer}. "
                f"Reply with just the final number.",
                reviewer_proxy,
            )
        )
        return verified

    return (solver_proxy, reviewer_proxy), invoke_fn, None


def run_model_selection(
    invoke_fn,
    llm_proxies,
    parallel=False,
    dataset_file=None,
    selector_name: str = "brute_force",
    sample_fraction: float = 0.25,
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    print(f"  [run] dataset loaded: {len(dataset)} samples from {dataset_file}")
    model_candidates = ["haiku", "sonnet"]
    mode = "parallel" if parallel else "sequential"
    print(f"  [run] starting model selection ({mode}) — candidates: {model_candidates}")

    kwargs = {
        "invoke_fn": invoke_fn,
    }

    if selector_name == "random_search":
        kwargs["sample_fraction"] = sample_fraction

    SelectorCls = SELECTORS[selector_name]
    selector = SelectorCls(
        models={llm: model_candidates for llm in llm_proxies},
        eval_fn=eval_fn,
        dataset=dataset,
        **kwargs,
    )

    results = selector.select_best(parallel=parallel)
    print(f"\nBest: {results.get_best()}")
    return results


def plot_results(results, title="Model Performance", save_path=None):
    """Plot accuracy vs latency for model selection results."""
    plt.figure(figsize=(10, 6))
    accuracies = [r.accuracy for r in results]
    latencies = [r.latency_seconds for r in results]
    names = [r.model_name for r in results]

    plt.scatter(latencies, accuracies)

    for name, lat, acc in zip(names, latencies, accuracies):
        plt.annotate(name, (lat, acc))

    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


EXAMPLES = {
    "single": ("Single-agent", single_agent_example),
    "multi": ("Multi-agent (shared LLM)", multi_agent_example),
    "multi-llm": ("Multi-agent (separate LLMs)", multi_agent_multi_llm_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Claude Agent SDK model selection example"
    )
    parser.add_argument(
        "example",
        choices=EXAMPLES.keys(),
        help="Which example to run",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run model selection in parallel",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math_problems.jsonl",
        help="JSONL filename in examples/datasets/ (default: math_problems.jsonl)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip saving the results plot"
    )
    parser.add_argument(
        "--selector",
        choices=sorted(SELECTORS.keys()),
        default="brute_force",
        help="Model selector to use (default: brute_force)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.25,
        help="Fraction of combinations to evaluate when --selector=random_search",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    mode = "parallel" if args.parallel else "sequential"

    print("=" * 40)
    print(f"Claude SDK — {label} ({mode})")
    print("=" * 40)

    print("\n[1] Setting up agents...")
    result = setup_fn()

    # All setup functions return 3-tuples: (proxy_or_proxies, invoke_fn, _)
    if isinstance(result[0], tuple):
        llm_proxies, invoke, _ = result
    else:
        llm_proxy, invoke, _ = result
        llm_proxies = [llm_proxy]

    print("\n[2] Running model selection...")
    results = run_model_selection(
        invoke,
        llm_proxies,
        parallel=args.parallel,
        dataset_file=args.dataset,
        selector_name=args.selector,
        sample_fraction=args.sample_fraction,
    )

    if not args.no_plot:
        print("\n[3] Saving results plot...")
        plot_results(
            results,
            f"Claude {label} Results",
            "examples/claude_results.png",
        )
