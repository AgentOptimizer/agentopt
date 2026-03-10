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

from agentopt import ModelProxy
from agentopt.model_selection import (
    BruteForceModelSelector,
    RandomSearchModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
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

    def clone_fn(model_map):
        """Build fresh agents for parallel evaluation.

        model_map: {proxy -> model_name_string} for this combination.
        Both agents share the same model since there is one proxy.
        """
        model_name = model_map[proxy]
        print(f"  [clone_fn] building fresh researcher + coder (model: {model_name})")
        new_config = LLMConfig(_build_config(model_name))
        fresh_researcher = ConversableAgent(
            name="researcher",
            system_message="You are a research assistant. Find and summarize information.",
            llm_config=new_config,
            human_input_mode="NEVER",
        )
        fresh_coder = ConversableAgent(
            name="coder",
            system_message="You are a coding assistant. Write efficient code based on research.",
            llm_config=new_config,
            human_input_mode="NEVER",
        )

        def fresh_invoke(input_data):
            question = input_data["input"]
            research_output = _extract(
                fresh_researcher.run(message=question, max_turns=1, user_input=False)
            )
            return _extract(
                fresh_coder.run(
                    message=f"Context: {research_output}\n\nTask: {question}",
                    max_turns=1,
                    user_input=False,
                )
            )

        return fresh_invoke

    return proxy, invoke_fn, clone_fn


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
        r_model = researcher_proxy.get_model().model
        c_model = coder_proxy.get_model().model
        print(
            f"  [invoke] researcher_model={r_model}, coder_model={c_model} | q: {question[:60]}"
        )
        research_output = _extract(
            researcher.run(message=question, max_turns=1, user_input=False)
        )
        print(f"  [invoke] coder processing with research context")
        coder_output = _extract(
            coder.run(
                message=f"Context: {research_output}\n\nTask: {question}",
                max_turns=1,
                user_input=False,
            )
        )
        return coder_output

    def clone_fn(model_map):
        """Build fresh agents for parallel evaluation.

        model_map: {researcher_proxy -> model_name, coder_proxy -> model_name}.
        Each proxy maps to its own model candidate for this combination.
        """
        r_model = model_map[researcher_proxy]
        c_model = model_map[coder_proxy]
        print(
            f"  [clone_fn] building fresh agents (researcher: {r_model}, coder: {c_model})"
        )
        fresh_researcher = ConversableAgent(
            name="researcher",
            system_message="You are a research assistant. Find and summarize information.",
            llm_config=LLMConfig(_build_config(r_model)),
            human_input_mode="NEVER",
        )
        fresh_coder = ConversableAgent(
            name="coder",
            system_message="You are a coding assistant. Write efficient code based on research.",
            llm_config=LLMConfig(_build_config(c_model)),
            human_input_mode="NEVER",
        )

        def fresh_invoke(input_data):
            question = input_data["input"]
            research_output = _extract(
                fresh_researcher.run(message=question, max_turns=1, user_input=False)
            )
            return _extract(
                fresh_coder.run(
                    message=f"Context: {research_output}\n\nTask: {question}",
                    max_turns=1,
                    user_input=False,
                )
            )

        return fresh_invoke

    return (researcher_proxy, coder_proxy), invoke_fn, clone_fn


def run_model_selection(
    agent_or_invoke_fn,
    llm_proxies,
    dataset_file=None,
    clone_fn=None,
    parallel: bool = False,
    selector_name: str = "brute_force",
    sample_fraction: float = 0.25,
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    print(f"  [run] dataset loaded: {len(dataset)} samples from {dataset_file}")
    model_candidates = ["gpt-4o-mini", "anthropic/claude-sonnet-4-20250514"]

    models = {p: model_candidates for p in llm_proxies}

    # Single agent: pass agent= so the selector auto-detects AG2's .run()
    # Multi-agent: pass invoke_fn= for custom chaining
    if callable(agent_or_invoke_fn) and not hasattr(agent_or_invoke_fn, "run"):
        kwargs = {"invoke_fn": agent_or_invoke_fn}
        if clone_fn is not None:
            kwargs["clone_fn"] = clone_fn
    else:
        kwargs = {"agent": agent_or_invoke_fn}

    if selector_name == "random_search":
        kwargs["sample_fraction"] = sample_fraction

    mode = "parallel" if parallel else "sequential"
    print(f"  [run] starting model selection ({mode}) — candidates: {model_candidates}")
    SelectorCls = SELECTORS[selector_name]
    selector = SelectorCls(
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
        help="Evaluate model combinations in parallel (requires clone_fn for multi examples)",
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
    print("=" * 40)
    print(f"AG2 — {label}")
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
        selector_name=args.selector,
        sample_fraction=args.sample_fraction,
    )

    if not args.no_plot:
        print("\n[3] Saving results plot...")
        plot_results(results, f"AG2 {label} Results", "examples/ag2_results.png")
