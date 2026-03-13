"""
AG2 (AutoGen 2) example for agentopt model selection.

Demonstrates single-agent, multi-agent (shared LLM), and multi-agent
(separate LLMs) setups using the native AG2 API.

ModelProxy wraps LLMConfig directly — the registration layer in
model_proxy/ag2.py handles the plumbing transparently.
"""

import argparse
import json
import os
from pathlib import Path

from autogen import ConversableAgent, LLMConfig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentopt import EvalCache, ModelProxy
from agentopt.model_selection import (
    BruteForceModelSelector,
    RandomSearchModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
    HyperbandModelSelector,
    BayesianOptimizationModelSelector,
)


def _build_config(model: str) -> dict:
    bare = model.split("/", 1)[-1] if "/" in model else model
    if bare.startswith("claude") or model.startswith("anthropic/"):
        return {
            "api_type": "anthropic",
            "model": bare,
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        }
    return {"api_type": "openai", "model": bare, "api_key": os.getenv("OPENAI_API_KEY")}


def _extract(response) -> str:
    for _ in response.events:
        pass
    if hasattr(response, "summary") and response.summary:
        return response.summary
    if response.messages:
        last = response.messages[-1]
        if isinstance(last, dict):
            return last.get("content") or str(last)
        return last.content if hasattr(last, "content") else str(last)
    return ""


SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random_search": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "hyperband": HyperbandModelSelector,
    "bayesian_optimization": BayesianOptimizationModelSelector,
}


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


def eval_fn(expected, actual):
    return expected.lower() in str(actual).lower()


def single_agent_example():
    """Single AG2 agent — one proxy, selector auto-detects .run()."""
    llm_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
    )
    proxy = ModelProxy(llm_config)
    print("  [setup] proxy created (initial model: gpt-4o-mini)")

    agent = ConversableAgent(
        name="math_assistant",
        system_message="You are a helpful math assistant. Answer questions concisely with the final numerical answer.",
        llm_config=proxy,
        human_input_mode="NEVER",
    )
    print("  [setup] agent created: math_assistant")

    # Single-agent uses agent= path; parallel is handled via AG2Adapter.clone_for_parallel.
    return proxy, agent, None


def multiagent_example():
    """Multi-agent (shared LLM) — researcher + coder with a single proxy."""
    llm_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
    )
    proxy = ModelProxy(llm_config)
    print("  [setup] proxy created (initial model: gpt-4o-mini)")

    researcher = ConversableAgent(
        name="researcher",
        system_message="You are a research assistant. Find and summarize information.",
        llm_config=proxy,
        human_input_mode="NEVER",
    )
    print("  [setup] agent created: researcher")

    coder = ConversableAgent(
        name="coder",
        system_message="You are a coding assistant. Write efficient code based on research.",
        llm_config=proxy,
        human_input_mode="NEVER",
    )
    print("  [setup] agent created: coder")
    print("  [setup] pipeline: researcher -> coder (shared proxy)")

    def invoke_fn(input_data):
        question = input_data["input"]
        model_name = proxy.get_model().model
        research_output = _extract(
            researcher.run(message=question, max_turns=1, user_input=False)
        )
        coder_output = _extract(
            coder.run(
                message=f"Context: {research_output}\n\nTask: {question}",
                max_turns=1,
                user_input=False,
            )
        )
        return coder_output

    return proxy, invoke_fn, None


def multiagent_multillm_example():
    """Multi-agent (separate LLMs) — independent proxy per agent."""
    researcher_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
    )
    coder_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
    )
    researcher_proxy = ModelProxy(researcher_config)
    print("  [setup] researcher_proxy created (initial model: gpt-4o-mini)")
    coder_proxy = ModelProxy(coder_config)
    print("  [setup] coder_proxy created (initial model: gpt-4o-mini)")

    researcher = ConversableAgent(
        name="researcher",
        system_message="You are a research assistant. Find and summarize information.",
        llm_config=researcher_proxy,
        human_input_mode="NEVER",
    )
    print("  [setup] agent created: researcher")

    coder = ConversableAgent(
        name="coder",
        system_message="You are a coding assistant. Write efficient code based on research.",
        llm_config=coder_proxy,
        human_input_mode="NEVER",
    )
    print("  [setup] agent created: coder")
    print("  [setup] pipeline: researcher (proxy1) -> coder (proxy2)")

    def invoke_fn(input_data):
        question = input_data["input"]
        research_output = _extract(
            researcher.run(message=question, max_turns=1, user_input=False)
        )
        coder_output = _extract(
            coder.run(
                message=f"Context: {research_output}\n\nTask: {question}",
                max_turns=1,
                user_input=False,
            )
        )
        return coder_output

    return (researcher_proxy, coder_proxy), invoke_fn, None


def run_model_selection(
    agent_or_invoke_fn,
    llm_proxies,
    dataset_file=None,
    parallel: bool = False,
    selector_name: str = "brute_force",
    selector_kwargs: dict | None = None,
    cache_path: str | None = None,
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    print(f"  [run] dataset loaded: {len(dataset)} samples from {dataset_file}")
    model_candidates = ["gpt-4o-mini", "gpt-5.1", "gpt-4o"]

    mode = "parallel" if parallel else "sequential"
    print(f"  [run] starting model selection ({mode}) — candidates: {model_candidates}")

    # Set up cache.
    cache = EvalCache(cache_path) if cache_path else None
    if cache:
        print(f"  [run] cache enabled: {cache_path} ({len(cache)} existing entries)")

    # Single agent: pass agent= so the selector auto-detects AG2's .run()
    # Multi-agent: pass invoke_fn= for custom chaining
    if callable(agent_or_invoke_fn) and not hasattr(agent_or_invoke_fn, "run"):
        agent_key, agent_val = "invoke_fn", agent_or_invoke_fn
    else:
        agent_key, agent_val = "agent", agent_or_invoke_fn

    SelectorCls = SELECTORS[selector_name]
    base_kwargs = {
        "models": {p: model_candidates for p in llm_proxies},
        "eval_fn": eval_fn,
        "dataset": dataset,
        agent_key: agent_val,
        "cache": cache,
    }
    if selector_kwargs:
        base_kwargs.update(selector_kwargs)
    selector = SelectorCls(**base_kwargs)
    results = selector.select_best(parallel=parallel)
    print(f"\nBest: {results.get_best()}")

    if cache:
        print(f"  [run] cache now has {len(cache)} entries")

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
    "multi": ("Multi-agent (shared LLM)", multiagent_example),
    "multi-llm": ("Multi-agent (separate LLMs)", multiagent_multillm_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AG2 model selection example")
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
    parser.add_argument(
        "--reduction-factor",
        type=float,
        default=3.0,
        help="Reduction factor η for hyperband selector (default: 3.0)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to cache file (e.g. .cache/ag2_eval.json). Omit to disable caching.",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    print("=" * 40)
    print(f"AG2 — {label}")
    print("=" * 40)

    print("\n[1] Setting up agents...")
    result = setup_fn()
    # All setup functions return (proxy_or_proxies, agent_or_invoke_fn, _).
    if isinstance(result[0], tuple):
        llm_proxies, agent_or_invoke, _ = result
    else:
        llm_proxy_or_agent, agent_or_invoke, _ = result
        llm_proxies = [llm_proxy_or_agent]

    print("\n[2] Running model selection...")
    selector_kwargs = {}
    if args.selector == "random_search":
        selector_kwargs["sample_fraction"] = args.sample_fraction
    if args.selector == "hyperband":
        selector_kwargs["reduction_factor"] = args.reduction_factor

    results = run_model_selection(
        agent_or_invoke,
        llm_proxies,
        dataset_file=args.dataset,
        parallel=args.parallel,
        selector_name=args.selector,
        selector_kwargs=selector_kwargs,
        cache_path=args.cache,
    )

    if not args.no_plot:
        print("\n[3] Saving results plot...")
        plot_results(results, f"AG2 {label} Results", "examples/ag2_results.png")
