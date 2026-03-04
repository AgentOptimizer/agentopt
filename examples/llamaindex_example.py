import argparse
import json
from pathlib import Path

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentopt import ModelProxy, BruteForceModelSelector
from agentopt.model_selection import (
    BaseModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
)
from agentopt.model_proxy.framework_specific_implementation import build_llamaindex_llm

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
}


def load_dataset(dataset_dir, filename):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for LlamaIndex."""
    dataset_path = Path(dataset_dir)
    jsonl_file = dataset_path / filename

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(({"user_msg": item["question"]}, item["output"]))
    return tasks


def eval_fn(expected, actual):
    """Check if expected answer appears in the agent's response."""
    return expected.lower() in str(actual).lower()


# Tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a and returns the result"""
    return a - b


def divide(a: float, b: float) -> float:
    """Divide a by b and returns the result"""
    if b == 0:
        return float("inf")
    return a / b


def single_agent_example():
    """Single LlamaIndex FunctionAgent wrapped in AgentWorkflow.

    LlamaIndex agents use strict Pydantic validation, so ModelProxy cannot
    be passed as llm= directly. Instead we create the agent with a real LLM.
    The selector auto-registers agents for sync on model swap.
    """
    initial_llm = build_llamaindex_llm("gpt-4o-mini")
    llm_proxy = ModelProxy(initial_llm)

    agent = FunctionAgent(
        name="MathAgent",
        description="Solves math problems using calculator tools",
        tools=[multiply, add, subtract, divide],
        llm=initial_llm,
        system_prompt=(
            "You are a helpful assistant that can perform mathematical operations. "
            "When asked to calculate something, use the available tools to compute the result."
        ),
    )

    workflow = AgentWorkflow(
        agents=[agent],
        root_agent="MathAgent",
    )

    return llm_proxy, workflow


def multi_agent_example():
    """Multi-agent LlamaIndex AgentWorkflow.

    Uses AgentWorkflow — LlamaIndex's native multi-agent orchestrator.
    Agent 1 (Math): Solves the problem using tools, hands off to Reviewer.
    Agent 2 (Reviewer): Verifies the answer and returns the final result.

    Shared LLM proxy so both agents are optimized together.
    """
    initial_llm = build_llamaindex_llm("gpt-4o-mini")
    llm_proxy = ModelProxy(initial_llm)

    math_agent = FunctionAgent(
        name="MathAgent",
        description="Solves math problems using calculator tools",
        tools=[multiply, add, subtract, divide],
        llm=initial_llm,
        system_prompt=(
            "You are a math assistant. Use the available tools to solve "
            "the calculation. Once you have the answer, hand off to "
            "ReviewAgent for verification."
        ),
        can_handoff_to=["ReviewAgent"],
    )

    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Reviews and verifies math answers",
        tools=[],
        llm=initial_llm,
        system_prompt=(
            "You are a reviewer. You receive a math question and a proposed "
            "answer. Verify the answer is correct and return the final numeric result. "
            "If the answer looks wrong, hand back to MathAgent. "
            "Reply with just the number."
        ),
        can_handoff_to=["MathAgent"],
    )

    workflow = AgentWorkflow(
        agents=[math_agent, review_agent],
        root_agent="MathAgent",
    )

    return llm_proxy, workflow


def multi_agent_multi_llm_example():
    """Multi-agent AgentWorkflow with separate LLMs per agent.

    Each agent has its own proxy for independent optimization.

    Note: Uses same-provider models to avoid cross-provider tool format
    issues in LlamaIndex AgentWorkflow (OpenAI and Anthropic use different
    tool_call serialization formats in conversation history).
    """
    math_llm = build_llamaindex_llm("claude-sonnet-4-20250514")
    reviewer_llm = build_llamaindex_llm("claude-sonnet-4-20250514")
    math_proxy = ModelProxy(math_llm)
    reviewer_proxy = ModelProxy(reviewer_llm)

    math_agent = FunctionAgent(
        name="MathAgent",
        description="Solves math problems using calculator tools",
        tools=[multiply, add, subtract, divide],
        llm=math_llm,
        system_prompt=(
            "You are a math assistant. Use the available tools to solve "
            "the calculation. Once you have the answer, hand off to "
            "ReviewAgent for verification."
        ),
        can_handoff_to=["ReviewAgent"],
    )

    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Reviews and verifies math answers",
        tools=[],
        llm=reviewer_llm,
        system_prompt=(
            "You are a reviewer. You receive a math question and a proposed "
            "answer. Verify the answer is correct and return the final numeric result. "
            "If the answer looks wrong, hand back to MathAgent. "
            "Reply with just the number."
        ),
        can_handoff_to=["MathAgent"],
    )

    workflow = AgentWorkflow(
        agents=[math_agent, review_agent],
        root_agent="MathAgent",
    )

    return (math_proxy, reviewer_proxy), workflow


def run_model_selection(
    agent,
    llm_proxies,
    parallel=False,
    dataset_file=None,
    model_candidates=None,
    selector_name: str = "brute_force",
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    if model_candidates is None:
        model_candidates = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]

    SelectorCls = SELECTORS[selector_name]
    selector = SelectorCls(
        models={llm: model_candidates for llm in llm_proxies},
        eval_fn=eval_fn,
        dataset=dataset,
        agent=agent,
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
    parser = argparse.ArgumentParser(description="LlamaIndex model selection example")
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
        help="JSONL filename in examples/datasets/ (default: first .jsonl found)",
    )
    parser.add_argument(
        "--selector",
        choices=sorted(SELECTORS.keys()),
        default="brute_force",
        help="Model selector to use (default: brute_force)",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    mode = "parallel" if args.parallel else "sequential"

    result = setup_fn()

    # multi-llm returns a tuple of proxies; the others return a single proxy
    if isinstance(result[0], tuple):
        llm_proxies, agent = result
    else:
        llm_proxy, agent = result
        llm_proxies = [llm_proxy]

    # Multi-LLM uses Anthropic-only candidates to avoid cross-provider
    # tool format issues in LlamaIndex AgentWorkflow.
    candidates = (
        ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]
        if args.example == "multi-llm"
        else None  # default mixed candidates
    )

    results = run_model_selection(
        agent,
        llm_proxies,
        parallel=args.parallel,
        dataset_file=args.dataset,
        model_candidates=candidates,
        selector_name=args.selector,
    )

    plot_results(
        results,
        f"LlamaIndex {label} Results",
        "examples/llamaindex_results.png",
    )
