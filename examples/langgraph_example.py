import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import SystemMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentopt import ModelProxy, BruteForceModelSelector


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
            review_prompt = SystemMessage(
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

    def invoke(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        return self.graph.invoke(input_state)


def multiagent_example():
    """Multi-agent (shared LLM) — solver + reviewer sharing one proxy."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    proxy = ModelProxy(llm)
    workflow = MultiAgentWorkflow(solver_model=proxy, reviewer_model=proxy)

    def invoke_fn(input_data):
        return workflow.invoke(input_data)

    return proxy, invoke_fn


def multiagent_multillm_example():
    """Multi-agent (separate LLMs) — independent proxy per node."""
    solver_llm = ChatOpenAI(model="gpt-4o-mini")
    reviewer_llm = ChatOpenAI(model="gpt-4o-mini")
    solver_proxy = ModelProxy(solver_llm)
    reviewer_proxy = ModelProxy(reviewer_llm)

    agent = MultiAgentWorkflow(solver_model=solver_proxy, reviewer_model=reviewer_proxy)

    def invoke_fn(input_data):
        return agent.invoke(input_data)

    return (solver_proxy, reviewer_proxy), invoke_fn


def run_model_selection(agent_or_invoke_fn, llm_proxies, dataset_file=None):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    model_candidates = ["openai/gpt-4o-mini", "openai/gpt-4o"]

    models = {p: model_candidates for p in llm_proxies}

    if callable(agent_or_invoke_fn) and not hasattr(agent_or_invoke_fn, "invoke"):
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
        "--no-plot",
        action="store_true",
        help="Skip saving the results plot",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]

    print("=" * 40)
    print(f"LangGraph — {label}")
    print("=" * 40)

    result = setup_fn()

    if isinstance(result[0], tuple):
        llm_proxies, agent_or_invoke = result
    else:
        llm_proxy, agent_or_invoke = result
        llm_proxies = [llm_proxy]

    results = run_model_selection(
        agent_or_invoke,
        llm_proxies,
        dataset_file=args.dataset,
    )

    if not args.no_plot:
        plot_results(
            results, f"LangGraph {label} Results", "examples/langgraph_results.png"
        )
