"""
OpenAI Agents SDK examples with AgentOpt model selection.

Demonstrates single-agent and multi-agent (separate LLMs) setups.
Each supports both sequential and parallel model selection.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents import Agent, Runner

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentopt import ModelProxy, BruteForceModelSelector
from agentopt.model_proxy.framework_specific_implementation.openai_sdk import (
    build_openai_agents_model,
)


def load_dataset(dataset_dir, filename):
    """Load JSONL dataset and return (input_data, expected_answer) tuples."""
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


def eval_fn(expected: str, actual: Any) -> bool:
    actual_text = actual.get("output") if isinstance(actual, dict) else str(actual)
    return expected.lower() in str(actual_text).lower()


def single_agent_example():
    """Single agent — one proxy, selector auto-detects via OpenAISDKAdapter."""
    proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    print("  [setup] proxy created (initial model: gpt-4o-mini)")

    agent = Agent(
        name="Math QA",
        model=proxy,
        instructions="Answer the user's math question concisely.",
    )
    print("  [setup] agent created: Math QA")

    return proxy, agent, None


def multillm_example():
    """Multi-agent (separate LLMs) — independent proxy per agent."""
    researcher_proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    print("  [setup] researcher_proxy created (initial model: gpt-4o-mini)")
    coder_proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    print("  [setup] coder_proxy created (initial model: gpt-4o-mini)")

    researcher = Agent(
        name="researcher",
        model=researcher_proxy,
        instructions="You are a research assistant. Analyze the question and provide relevant context and findings.",
    )
    print("  [setup] agent created: researcher")

    coder = Agent(
        name="coder",
        model=coder_proxy,
        instructions="You are a coding assistant. Based on the provided context, give a concise final answer.",
    )
    print("  [setup] agent created: coder")
    print("  [setup] pipeline: researcher (proxy1) -> coder (proxy2)")

    def invoke_fn(input_data):
        question = input_data["input"]
        r_model = researcher_proxy.get_model().model
        c_model = coder_proxy.get_model().model
        print(f"  [invoke] researcher={r_model}, coder={c_model} | q: {question[:60]}")
        research_result = Runner.run_sync(researcher, question)
        research_output = (
            research_result.final_output
            if hasattr(research_result, "final_output")
            else str(research_result)
        )
        coder_result = Runner.run_sync(
            coder, f"Context: {research_output}\n\nTask: {question}"
        )
        return (
            coder_result.final_output
            if hasattr(coder_result, "final_output")
            else str(coder_result)
        )

    def clone_fn(model_map):
        """Build fresh agents for parallel evaluation.

        model_map: {researcher_proxy -> model_name, coder_proxy -> model_name}.
        """
        r_model = model_map[researcher_proxy]
        c_model = model_map[coder_proxy]
        print(
            f"  [clone_fn] building fresh agents (researcher: {r_model}, coder: {c_model})"
        )
        fresh_researcher = Agent(
            name="researcher",
            model=build_openai_agents_model(r_model),
            instructions="You are a research assistant. Analyze the question and provide relevant context and findings.",
        )
        fresh_coder = Agent(
            name="coder",
            model=build_openai_agents_model(c_model),
            instructions="You are a coding assistant. Based on the provided context, give a concise final answer.",
        )

        def fresh_invoke(input_data):
            question = input_data["input"]
            research_result = Runner.run_sync(fresh_researcher, question)
            research_output = (
                research_result.final_output
                if hasattr(research_result, "final_output")
                else str(research_result)
            )
            coder_result = Runner.run_sync(
                fresh_coder, f"Context: {research_output}\n\nTask: {question}"
            )
            return (
                coder_result.final_output
                if hasattr(coder_result, "final_output")
                else str(coder_result)
            )

        return fresh_invoke

    return (researcher_proxy, coder_proxy), invoke_fn, clone_fn


def run_model_selection(
    agent_or_invoke_fn, llm_proxies, dataset_file=None, clone_fn=None, parallel=False
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    print(f"  [run] dataset loaded: {len(dataset)} samples from {dataset_file}")
    model_candidates = ["gpt-4o-mini", "gpt-4o"]

    models = {p: model_candidates for p in llm_proxies}

    # Single agent: pass agent= so the selector auto-detects via OpenAISDKAdapter
    # Multi-agent: pass invoke_fn= for custom chaining
    if callable(agent_or_invoke_fn) and not hasattr(agent_or_invoke_fn, "name"):
        kwargs = {"invoke_fn": agent_or_invoke_fn}
        if clone_fn is not None:
            kwargs["clone_fn"] = clone_fn
    else:
        kwargs = {"agent": agent_or_invoke_fn}

    mode = "parallel" if parallel else "sequential"
    print(f"  [run] starting model selection ({mode}) — candidates: {model_candidates}")
    selector = BruteForceModelSelector(
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        **kwargs,
    )
    results = selector.select_best(parallel=parallel)
    print(f"\nBest: {results.get_best()}")
    return results


def plot_results(results, title="Model Performance", save_path=None):
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
    "multi-llm": ("Multi-agent (separate LLMs)", multillm_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenAI Agents SDK model selection example"
    )
    parser.add_argument("example", choices=EXAMPLES.keys(), help="Which example to run")
    parser.add_argument(
        "--dataset",
        type=str,
        default="math_problems.jsonl",
        help="JSONL filename in examples/datasets/",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate model combinations in parallel",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip saving the results plot"
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    print("=" * 40)
    print(f"OpenAI SDK — {label}")
    print("=" * 40)

    print("\n[1] Setting up agents...")
    result = setup_fn()
    # All setup functions return (proxy_or_proxies, agent_or_invoke_fn, clone_fn).
    if isinstance(result[0], tuple):
        llm_proxies, agent_or_invoke, clone_fn = result
    else:
        llm_proxy_or_agent, agent_or_invoke, clone_fn = result
        llm_proxies = [llm_proxy_or_agent]

    print("\n[2] Running model selection...")
    results = run_model_selection(
        agent_or_invoke,
        llm_proxies,
        dataset_file=args.dataset,
        clone_fn=clone_fn,
        parallel=args.parallel,
    )

    if not args.no_plot:
        print("\n[3] Saving results plot...")
        plot_results(
            results, f"OpenAI SDK {label} Results", "examples/openai_sdk_results.png"
        )
