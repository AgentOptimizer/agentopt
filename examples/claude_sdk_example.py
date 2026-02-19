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

from agentopt import ModelProxy, BruteForceModelSelector


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
    proxy = ModelProxy(ClaudeAgentOptions(model="claude-3-5-haiku-latest"))

    # BruteForceModelSelector swaps the model via proxy.set_model() before
    # calling invoke_fn, so the proxy is captured here via closure.
    def invoke_fn(input_data):
        return asyncio.run(_query_async(input_data["input"], proxy))

    return proxy, invoke_fn


def multi_agent_example():
    """Multi-agent chain with a shared LLM.

    Solver answers the question, then Reviewer verifies the answer.
    Both use the same ModelProxy so they are optimized together.
    """
    proxy = ModelProxy(ClaudeAgentOptions(model="claude-3-5-haiku-latest"))

    # BruteForceModelSelector swaps the model via proxy.set_model() before
    # calling invoke_fn, so the proxy is captured here via closure.
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

    return proxy, invoke_fn


def multi_agent_multi_llm_example():
    """Multi-agent chain with separate LLMs per agent.

    Solver and Reviewer each have their own ModelProxy so they can be
    optimized independently by BruteForceModelSelector.
    """
    solver_proxy = ModelProxy(ClaudeAgentOptions(model="claude-3-5-haiku-latest"))
    reviewer_proxy = ModelProxy(ClaudeAgentOptions(model="claude-3-5-haiku-latest"))

    # BruteForceModelSelector swaps the model via proxy.set_model() before
    # calling invoke_fn, so proxies are captured here via closure.
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

    return (solver_proxy, reviewer_proxy), invoke_fn


def run_model_selection(invoke_fn, llm_proxies, parallel=False, dataset_file=None):
    dataset = load_dataset("examples/datasets", filename=dataset_file)

    selector = BruteForceModelSelector(
        models={
            llm: [
                "claude-3-5-haiku-latest",
                "claude-sonnet-4-20250514",
            ]
            for llm in llm_proxies
        },
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
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
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    mode = "parallel" if args.parallel else "sequential"

    print("=" * 40)
    print(f"{label} ({mode})")
    print("=" * 40)

    result = setup_fn()

    # multi-llm returns a tuple of proxies; the others return a single proxy
    if isinstance(result[0], tuple):
        llm_proxies, invoke = result
    else:
        llm_proxy, invoke = result
        llm_proxies = [llm_proxy]

    results = run_model_selection(
        invoke,
        llm_proxies,
        parallel=args.parallel,
        dataset_file=args.dataset,
    )

    plot_results(
        results,
        f"Claude {label} Results",
        "examples/claude_results.png",
    )
