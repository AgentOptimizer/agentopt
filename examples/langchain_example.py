import argparse
import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
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

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random_search": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "hyperband": HyperbandModelSelector,
    "bayesian_optimization": BayesianOptimizationModelSelector,
}


def load_dataset(dataset_dir, filename):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for LangChain."""
    dataset_path = Path(dataset_dir)
    jsonl_file = dataset_path / filename

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Format input for LangChain's invoke({"input": ...})
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks


def eval_fn(expected, actual):
    return expected.lower() in str(actual.get("output", "")).lower()


# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2' or '10 * 5'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def single_agent_example():
    """Single agent with search and calculator tools."""
    # 1. Wrap the LLM
    llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
    print("  [setup] proxy created (initial model: gpt-4o-mini)")

    # 2. Define tools
    search = DuckDuckGoSearchRun()
    tools = [search, calculator]
    print("  [setup] tools defined: DuckDuckGoSearch, calculator")

    # 3. Create agent prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful research assistant. Use tools when needed."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    print("  [setup] prompt created")

    # 4. Create agent and executor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    print("  [setup] agent executor created: single researcher agent")

    return llm, agent_executor


def run_model_selection(
    agent_or_invoke_fn,
    llm_proxies,
    parallel=False,
    dataset_file=None,
    selector_name: str = "brute_force",
    selector_kwargs: dict | None = None,
    cache_path: str | None = None,
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    print(f"  [run] dataset loaded: {len(dataset)} samples from {dataset_file}")
    model_candidates = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
    ]
    mode = "parallel" if parallel else "sequential"
    print(f"  [run] starting model selection ({mode}) — candidates: {model_candidates}")

    # Set up cache.
    cache = EvalCache(cache_path) if cache_path else None
    if cache:
        print(f"  [run] cache enabled: {cache_path} ({len(cache)} existing entries)")

    SelectorCls = SELECTORS[selector_name]
    base_kwargs = {
        "models": {
            llm: [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
            ]
            for llm in llm_proxies
        },
        "eval_fn": eval_fn,
        "dataset": dataset,
        "agent": agent_or_invoke_fn,
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


def multiagent_example():
    raise SystemExit("Multi-agent (shared LLM) is not available for LangChain example.")


def multiagent_multillm_example():
    raise SystemExit(
        "Multi-agent (separate LLMs) is not available for LangChain example."
    )


EXAMPLES = {
    "single": ("Single-agent", single_agent_example),
    "multi": ("Multi-agent (shared LLM)", multiagent_example),
    "multi-llm": ("Multi-agent (separate LLMs)", multiagent_multillm_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain model selection example")
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
        help="Path to cache file (e.g. .cache/langchain_eval.json). Omit to disable caching.",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    mode = "parallel" if args.parallel else "sequential"

    print("=" * 40)
    print(f"{label} ({mode})")
    print("=" * 40)

    print("\n[1] Setting up agents...")
    result = setup_fn()
    # multi-llm returns a tuple of proxies; the others return a single proxy
    if isinstance(result[0], tuple):
        llm_proxies, agent_executor = result
    else:
        llm_proxy, agent_executor = result
        llm_proxies = [llm_proxy]

    print("\n[2] Running model selection...")
    selector_kwargs = {}
    if args.selector == "random_search":
        selector_kwargs["sample_fraction"] = args.sample_fraction
    if args.selector == "hyperband":
        selector_kwargs["reduction_factor"] = args.reduction_factor

    results = run_model_selection(
        agent_executor,
        llm_proxies,
        parallel=args.parallel,
        dataset_file=args.dataset,
        selector_name=args.selector,
        selector_kwargs=selector_kwargs,
        cache_path=args.cache,
    )

    print("\n[3] Saving results plot...")
    plot_results(
        results, f"LangChain {label} Results", "examples/langchain_results.png"
    )
