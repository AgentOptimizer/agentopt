import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

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


class WorkflowState(TypedDict):
    messages: List[Any]
    final: str


def _normalize_messages(messages: List[Any]) -> List[Any]:
    return list(convert_to_messages(messages))


def _is_system_message(message: Any) -> bool:
    if isinstance(message, dict):
        return message.get("role") == "system"
    if hasattr(message, "type"):
        return message.type == "system"
    return False


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        if "final" in result:
            return str(result["final"])
        if "messages" in result and result["messages"]:
            last = result["messages"][-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
            if hasattr(last, "content"):
                return str(last.content)
            return str(last)
    return str(result)


def load_dataset(dataset_dir, filename):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for LangGraph."""
    dataset_path = Path(dataset_dir)
    jsonl_file = dataset_path / filename

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(
                (
                    {
                        "messages": [
                            {"role": "user", "content": item["question"]},
                        ]
                    },
                    item["output"],
                )
            )
    return tasks


def eval_fn(expected: str, actual: Any) -> bool:
    return expected.lower() in _extract_text(actual).lower()


class MultiAgentWorkflow:
    """
    Two-step LangGraph workflow: solver answers, reviewer verifies.
    Both nodes use the same model reference (can be shared or separate proxies).
    """

    def __init__(self, solver_model: Any, reviewer_model: Any) -> None:
        self.solver_model = solver_model
        self.reviewer_model = reviewer_model
        self.graph = self._build_graph()

    def _build_graph(self):
        def add_system(state: WorkflowState) -> WorkflowState:
            messages = _normalize_messages(list(state.get("messages", [])))
            if messages and _is_system_message(messages[0]):
                return {"messages": messages, "final": state.get("final", "")}
            return {
                "messages": [SystemMessage(content="You are a helpful math assistant.")]
                + messages,
                "final": state.get("final", ""),
            }

        def call_solver(state: WorkflowState) -> WorkflowState:
            messages = _normalize_messages(list(state.get("messages", [])))
            response = self.solver_model.invoke(messages)
            return {
                "messages": messages + [response],
                "final": state.get("final", ""),
            }

        def call_reviewer(state: WorkflowState) -> WorkflowState:
            messages = _normalize_messages(list(state.get("messages", [])))
            review_prompt = HumanMessage(
                content=(
                    "Review the previous answer. If correct, repeat the final "
                    "answer. If wrong, provide the corrected answer."
                )
            )
            messages_with_review = messages + [review_prompt]
            response = self.reviewer_model.invoke(messages_with_review)
            return {
                "messages": messages + [response],
                "final": state.get("final", ""),
            }

        def finalize(state: WorkflowState) -> WorkflowState:
            messages = list(state.get("messages", []))
            last = messages[-1] if messages else ""
            if isinstance(last, dict):
                final = str(last.get("content", "")).strip()
            elif hasattr(last, "content"):
                final = str(last.content).strip()
            else:
                final = str(last).strip()
            return {"messages": messages, "final": final}

        graph = StateGraph(WorkflowState)
        graph.add_node("add_system", add_system)
        graph.add_node("call_solver", call_solver)
        graph.add_node("call_reviewer", call_reviewer)
        graph.add_node("finalize", finalize)
        graph.set_entry_point("add_system")
        graph.add_edge("add_system", "call_solver")
        graph.add_edge("call_solver", "call_reviewer")
        graph.add_edge("call_reviewer", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def invoke(
        self, input_state: Dict[str, Any], config=None, **kwargs
    ) -> Dict[str, Any]:
        return self.graph.invoke(input_state, config, **kwargs)


def multiagent_example():
    """Multi-agent (shared LLM) — solver + reviewer sharing one proxy."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    proxy = ModelProxy(llm)
    print("  [setup] proxy created (initial model: gpt-4o-mini)")

    workflow = MultiAgentWorkflow(solver_model=proxy, reviewer_model=proxy)
    print(
        "  [setup] LangGraph workflow compiled: add_system -> call_solver -> call_reviewer -> finalize"
    )
    print("  [setup] both nodes share the same proxy")

    def invoke_fn(input_data):
        return workflow.invoke(input_data)

    return proxy, invoke_fn, None


def multiagent_multillm_example():
    """Multi-agent (separate LLMs) — independent proxy per node.

    Solver uses OpenAI, reviewer uses Anthropic ().
    """
    solver_llm = ChatOpenAI(model="gpt-4o-mini")
    reviewer_llm = ChatAnthropic(model="got-4.1")
    solver_proxy = ModelProxy(solver_llm)
    print("  [setup] solver_proxy created (initial model: gpt-4o-mini)")
    reviewer_proxy = ModelProxy(reviewer_llm)
    print("  [setup] reviewer_proxy created (initial model: gpt-4.1)")

    agent = MultiAgentWorkflow(solver_model=solver_proxy, reviewer_model=reviewer_proxy)
    print(
        "  [setup] LangGraph workflow compiled: add_system -> call_solver (OpenAI) -> call_reviewer () -> finalize"
    )

    def invoke_fn(input_data):
        return agent.invoke(input_data)

    return (solver_proxy, reviewer_proxy), invoke_fn, None


def run_model_selection(
    agent_or_invoke_fn,
    llm_proxies,
    dataset_file=None,
    parallel=False,
    model_candidates=None,
    selector_name: str = "brute_force",
    selector_kwargs: dict | None = None,
    cache_path: str | None = None,
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    print(f"  [run] dataset loaded: {len(dataset)} samples from {dataset_file}")
    if model_candidates is None:
        model_candidates = {
            p: ["openai/gpt-4o-mini", "openai/gpt-4o"] for p in llm_proxies
        }
    elif isinstance(model_candidates, list):
        model_candidates = {p: model_candidates for p in llm_proxies}

    models = model_candidates

    # Set up cache.
    cache = EvalCache(cache_path) if cache_path else None
    if cache:
        print(f"  [run] cache enabled: {cache_path} ({len(cache)} existing entries)")

    if callable(agent_or_invoke_fn) and not hasattr(agent_or_invoke_fn, "invoke"):
        agent_key, agent_val = "invoke_fn", agent_or_invoke_fn
    else:
        agent_key, agent_val = "agent", agent_or_invoke_fn

    SelectorCls = SELECTORS[selector_name]
    base_kwargs = {
        "models": models,
        "eval_fn": eval_fn,
        "dataset": dataset,
        agent_key: agent_val,
        "cache": cache,
    }
    if selector_kwargs:
        base_kwargs.update(selector_kwargs)
    if selector_name == "hyperband":
        base_kwargs["reduction_factor"] = reduction_factor
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


EXAMPLES = {
    "multi": ("Multi-agent (shared LLM)", multiagent_example),
    "multi-llm": ("Multi-agent (separate LLMs)", multiagent_multillm_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph model selection example")
    parser.add_argument(
        "example",
        choices=EXAMPLES.keys(),
        help="Which example to run",
    )
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
        "--no-plot",
        action="store_true",
        help="Skip saving the results plot",
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
        help="Path to cache file (e.g. .cache/langgraph_eval.json). Omit to disable caching.",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]

    print("=" * 40)
    print(f"LangGraph — {label}")
    print("=" * 40)

    print("\n[1] Setting up agents...")
    result = setup_fn()

    # All setup functions return (proxy_or_proxies, invoke_fn, _).
    if isinstance(result[0], tuple):
        llm_proxies, agent_or_invoke, _ = result
    else:
        llm_proxy, agent_or_invoke, _ = result
        llm_proxies = [llm_proxy]

    # Per-proxy candidates for multi-llm: both proxies search the same 2 models
    # → 4 combos: (gpt-4o, gpt-4o), (gpt-4o, -haiku), (-haiku, gpt-4o), (-haiku, -haiku)
    per_proxy_candidates = None
    if args.example == "multi-llm" and len(llm_proxies) == 2:
        candidates = ["openai/gpt-4o", "openai/gpt-4.1"]
        per_proxy_candidates = {p: candidates for p in llm_proxies}

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
        model_candidates=per_proxy_candidates,
        selector_name=args.selector,
        selector_kwargs=selector_kwargs,
        cache_path=args.cache,
    )

    if not args.no_plot:
        print("\n[3] Saving results plot...")
        plot_results(
            results, f"LangGraph {label} Results", "examples/langgraph_results.png"
        )
