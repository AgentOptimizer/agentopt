"""
AG2 (AutoGen 2) example for agentopt model selection.

Uses ModelProxy with LLMConfig and custom invoke_fn, since AG2 agents
use run()/process() rather than invoke()/kickoff().
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
import matplotlib.pyplot as plt

from agentopt import ModelProxy, BruteForceModelSelector

load_dotenv()


def load_dataset(dataset_dir):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for AG2."""
    dataset_path = Path(dataset_dir)
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in: {dataset_dir}")

    tasks = []
    with open(jsonl_files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks


def eval_fn(expected, actual):
    return expected.lower() in str(actual).lower()


def make_llm_config(model: str) -> LLMConfig:
    """Create an LLMConfig for the given model."""
    return LLMConfig(
        config_list=[
            {
                "api_type": "openai",
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ]
    )


class AG2ConfigWrapper:
    """
    Mutable wrapper for AG2 LLMConfig. AG2 validates llm_config and rejects ModelProxy,
    so we use a real LLMConfig backed by mutable config_list. ModelProxy wraps this
    wrapper and updates config_list[0]['model'] when set_model() is called.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.config_list = [
            {
                "api_type": "openai",
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ]

    @property
    def model(self) -> str:
        return self.config_list[0]["model"]

    @model.setter
    def model(self, value: str) -> None:
        self.config_list[0]["model"] = value


def single_agent_example():
    """Single agent with proxied LLMConfig."""
    # 1. Use mutable config: AG2 rejects ModelProxy, so we pass a real LLMConfig
    #    backed by config_list that ModelProxy (via AG2ConfigWrapper) can update
    config_wrapper = AG2ConfigWrapper("gpt-4o-mini")
    llm_config = LLMConfig(config_list=config_wrapper.config_list)
    llm_config_proxy = ModelProxy(config_wrapper)

    # 2. Create agent with real LLMConfig (AG2 validates this)
    agent = ConversableAgent(
        name="math_assistant",
        system_message="You are a helpful math assistant. Answer questions concisely with the final numerical answer.",
        llm_config=llm_config,  # real LLMConfig; proxy updates config_list in place
        human_input_mode="NEVER",
    )

    # 3. Custom invoke_fn: AG2 uses run()/process(), not invoke/kickoff
    def invoke_fn(input_data):
        response = agent.run(
            message=input_data["input"],
            max_turns=2,
            user_input=False,
        )
        # Iterate to execute (process() prints to console, so we iterate manually)
        for _ in response.events:
            pass
        # Extract result from summary or last message
        if hasattr(response, "summary") and response.summary:
            return response.summary
        if response.messages:
            last_msg = response.messages[-1]
            content = getattr(last_msg, "content", None) or str(last_msg)
            return content
        return ""

    # proxy updates config_wrapper.config_list[0]["model"] when set_model() is called
    return llm_config_proxy, invoke_fn


def multiagent_example():
    """
    Multi-agent setup with different LLMs for different agents.
    Uses researcher + coder pattern with custom invoke_fn.
    """
    # 1. Use mutable config wrappers (AG2 rejects ModelProxy as llm_config)
    config_research = AG2ConfigWrapper("gpt-4o-mini")
    config_coder = AG2ConfigWrapper("gpt-4o-mini")
    llm_research = ModelProxy(config_research)
    llm_coder = ModelProxy(config_coder)

    # 2. Create researcher agent with real LLMConfig
    researcher = ConversableAgent(
        name="researcher",
        system_message="You are a research assistant. Find and summarize information.",
        llm_config=LLMConfig(config_list=config_research.config_list),
        human_input_mode="NEVER",
    )

    # 3. Create coder agent with real LLMConfig
    coder = ConversableAgent(
        name="coder",
        system_message="You are a coding assistant. Write efficient code based on research.",
        llm_config=LLMConfig(config_list=config_coder.config_list),
        human_input_mode="NEVER",
    )

    # 4. Custom invoke_fn: chain agents sequentially
    def chained_invoke(input_data):
        question = input_data["input"]
        # Researcher runs first
        research_response = researcher.run(
            message=question,
            max_turns=2,
            user_input=False,
        )
        for _ in research_response.events:
            pass
        research_output = ""
        if research_response.messages:
            last = research_response.messages[-1]
            research_output = getattr(last, "content", "") or str(last)

        # Coder runs with research context
        coder_response = coder.run(
            message=f"Context: {research_output}\n\nTask: {question}",
            max_turns=2,
            user_input=False,
        )
        for _ in coder_response.events:
            pass
        if coder_response.messages:
            last = coder_response.messages[-1]
            return getattr(last, "content", "") or str(last)
        return ""

    return (llm_research, llm_coder), chained_invoke


def run_model_selection(invoke_fn, llm_proxies, model_candidates=None):
    dataset = load_dataset("examples/datasets")

    if model_candidates is None:
        model_candidates = ["gpt-4o-mini", "gpt-4.1-mini"]

    # ModelProxy wraps AG2ConfigWrapper; set_model() accepts strings (updates model in place)
    models = {
        proxy: model_candidates
        for proxy in llm_proxies
    }

    selector = BruteForceModelSelector(
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
    )

    results = selector.select_best()
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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AG2 model selection example")
    parser.add_argument(
        "mode",
        choices=["single", "multi"],
        help="Run single-agent or multi-agent example",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip showing/saving the results plot",
    )
    args = parser.parse_args()

    if args.mode == "single":
        print("=" * 20)
        print("Single-agent example")
        print("=" * 20)
        llm_proxy, invoke_fn = single_agent_example()
        results = run_model_selection(invoke_fn, [llm_proxy])
    else:
        print("=" * 20)
        print("Multi-agent example")
        print("=" * 20)
        llm_proxies, chained_invoke = multiagent_example()
        results = run_model_selection(chained_invoke, llm_proxies)

    if not args.no_plot:
        plot_results(
            results, "AG2 Model Selection Results", "examples/ag2_results.png"
        )
