"""
AG2 (AutoGen 2) example for agentopt model selection.

Uses the model_proxy AG2 backend: create proxy and invoke_fn via ag2 helpers,
then run BruteForceModelSelector. All AG2/LLMConfig details live in model_proxy/ag2.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentopt import BruteForceModelSelector
from agentopt.model_proxy import ag2


def load_dataset(dataset_dir: str, filename: str = "math_problems.jsonl"):
    """Load JSONL dataset: list of (input_data, expected_answer) for AG2."""
    path = Path(dataset_dir) / filename
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks


def eval_fn(expected: str, actual: str) -> bool:
    return expected.lower() in str(actual).lower()


def single_agent_example():
    """Single agent with proxied LLM — one proxy, one invoke_fn."""
    return ag2.create_single_agent_proxy_and_invoke(default_model="gpt-4o-mini")


def multi_agent_example():
    """Multi-agent (researcher + coder) with separate proxies per agent."""
    return ag2.create_multi_agent_proxies_and_invoke(default_model="gpt-4o-mini")


def run_model_selection(invoke_fn, llm_proxies: list, model_candidates: list | None = None):
    dataset = load_dataset("examples/datasets")
    if model_candidates is None:
        model_candidates = ["gpt-4o-mini", "gpt-4.1-mini"]

    models = {p: model_candidates for p in llm_proxies}
    selector = BruteForceModelSelector(
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
    )
    results = selector.select_best()
    print(f"\nBest: {results.get_best()}")
    return results


def plot_results(results, title: str = "Model Performance", save_path: str | None = None):
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
    "multi": ("Multi-agent (researcher + coder)", multi_agent_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AG2 model selection example")
    parser.add_argument("mode", choices=EXAMPLES.keys(), help="single or multi-agent")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the results plot")
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.mode]
    print("=" * 40)
    print(f"AG2 — {label}")
    print("=" * 40)

    result = setup_fn()
    if isinstance(result[0], tuple):
        llm_proxies, invoke_fn = result
    else:
        llm_proxy, invoke_fn = result
        llm_proxies = [llm_proxy]

    results = run_model_selection(invoke_fn, llm_proxies)

    if not args.no_plot:
        plot_results(results, f"AG2 {label} Results", "examples/ag2_results.png")
