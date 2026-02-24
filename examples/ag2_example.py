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

from agentopt import ModelProxy, BruteForceModelSelector
from agentopt.model_proxy.ag2 import extract_ag2_content


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

    agent = ConversableAgent(
        name="math_assistant",
        system_message="You are a helpful math assistant. Answer questions concisely with the final numerical answer.",
        llm_config=proxy,
        human_input_mode="NEVER",
    )

    return proxy, agent


def multiagent_example():
    """Multi-agent (shared LLM) — researcher + coder with a single proxy."""
    llm_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
    )
    proxy = ModelProxy(llm_config)

    researcher = ConversableAgent(
        name="researcher",
        system_message="You are a research assistant. Find and summarize information.",
        llm_config=proxy,
        human_input_mode="NEVER",
    )

    coder = ConversableAgent(
        name="coder",
        system_message="You are a coding assistant. Write efficient code based on research.",
        llm_config=proxy,
        human_input_mode="NEVER",
    )

    def invoke_fn(input_data):
        question = input_data["input"]
        research_output = extract_ag2_content(
            researcher.run(message=question, max_turns=2, user_input=False)
        )
        return extract_ag2_content(
            coder.run(
                message=f"Context: {research_output}\n\nTask: {question}",
                max_turns=2,
                user_input=False,
            )
        )

    return proxy, invoke_fn


def multiagent_multillm_example():
    """Multi-agent (separate LLMs) — independent proxy per agent."""
    researcher_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
    )
    coder_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
    )
    researcher_proxy = ModelProxy(researcher_config)
    coder_proxy = ModelProxy(coder_config)

    researcher = ConversableAgent(
        name="researcher",
        system_message="You are a research assistant. Find and summarize information.",
        llm_config=researcher_proxy,
        human_input_mode="NEVER",
    )

    coder = ConversableAgent(
        name="coder",
        system_message="You are a coding assistant. Write efficient code based on research.",
        llm_config=coder_proxy,
        human_input_mode="NEVER",
    )

    def invoke_fn(input_data):
        question = input_data["input"]
        research_output = extract_ag2_content(
            researcher.run(message=question, max_turns=2, user_input=False)
        )
        return extract_ag2_content(
            coder.run(
                message=f"Context: {research_output}\n\nTask: {question}",
                max_turns=2,
                user_input=False,
            )
        )

    return (researcher_proxy, coder_proxy), invoke_fn


def run_model_selection(agent_or_invoke_fn, llm_proxies, dataset_file=None):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    model_candidates = ["gpt-4o-mini", "gpt-4.1-mini"]

    models = {p: model_candidates for p in llm_proxies}

    # Single agent: pass agent= so the selector auto-detects AG2's .run()
    # Multi-agent: pass invoke_fn= for custom chaining
    if callable(agent_or_invoke_fn) and not hasattr(agent_or_invoke_fn, "run"):
        kwargs = {"invoke_fn": agent_or_invoke_fn}
    else:
        kwargs = {"agent": agent_or_invoke_fn}

    selector = BruteForceModelSelector(
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        **kwargs,
    )
    results = selector.select_best()
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
        "--no-plot", action="store_true", help="Skip saving the results plot"
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    print("=" * 40)
    print(f"AG2 — {label}")
    print("=" * 40)

    result = setup_fn()
    # multi-llm returns a tuple of proxies; the others return a single proxy or agent
    if isinstance(result[0], tuple):
        llm_proxies, agent_or_invoke = result
    else:
        llm_proxy_or_agent, agent_or_invoke = result
        llm_proxies = [llm_proxy_or_agent]

    results = run_model_selection(
        agent_or_invoke,
        llm_proxies,
        dataset_file=args.dataset,
    )

    if not args.no_plot:
        plot_results(results, f"AG2 {label} Results", "examples/ag2_results.png")
