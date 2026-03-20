"""
Example: LangGraph agent with agentopt.

Prerequisites:
    1. pip install langchain-openai langgraph agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import inspect
from typing import Annotated, Any, Dict, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agentopt import (
    ArmEliminationModelSelector,
    BruteForceModelSelector,
    EpsilonLUCBModelSelector,
    HillClimbingModelSelector,
    LMProposalModelSelector,
    RandomSearchModelSelector,
    ThresholdBanditSEModelSelector,
)

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "epsilon_lucb": EpsilonLUCBModelSelector,
    "threshold_successive_elimination": ThresholdBanditSEModelSelector,
    "lm_proposal": LMProposalModelSelector,
}

try:
    from agentopt import BayesianOptimizationModelSelector

    SELECTORS["bayesian_optimization"] = BayesianOptimizationModelSelector
except ImportError:
    pass


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    answer: str


def agent_maker(models: Dict[str, Any]):
    """Factory: builds a LangGraph planner+solver agent."""
    planner_llm = (
        models["planner"]
        if not isinstance(models["planner"], str)
        else ChatOpenAI(model=models["planner"])
    )
    solver_llm = (
        models["solver"]
        if not isinstance(models["solver"], str)
        else ChatOpenAI(model=models["solver"])
    )

    def planner_node(state: AgentState) -> dict:
        response = planner_llm.invoke(
            [
                {
                    "role": "system",
                    "content": "Create a brief plan to answer the question.",
                }
            ]
            + state["messages"]
        )
        return {"plan": response.content}

    def solver_node(state: AgentState) -> dict:
        response = solver_llm.invoke(
            [
                {
                    "role": "system",
                    "content": f"Follow this plan and answer concisely:\n{state['plan']}",
                },
                state["messages"][-1],
            ]
        )
        return {"answer": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("solver", solver_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "solver")
    graph.add_edge("solver", END)
    app = graph.compile()

    # Return a callable that takes input and returns the answer
    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]
        result = app.invoke({"messages": [{"role": "user", "content": question}]})
        return result["answer"]

    return run


def eval_fn(expected: str, actual) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
]


def _filter_selector_kwargs(
    selector_cls, selector_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    params = inspect.signature(selector_cls.__init__).parameters
    return {k: v for k, v in selector_kwargs.items() if k in params}


def main():
    parser = argparse.ArgumentParser(description="LangGraph model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument(
        "--use-instances",
        action="store_true",
        help="Pass pre-built ChatOpenAI instances instead of model name strings",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.25,
        help="Fraction of combinations to evaluate when --selector=random",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Batch size for batched selectors "
            "(hill_climbing neighbours and bayesian_optimization candidates)."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Epsilon for --selector=epsilon_lucb.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for --selector=threshold_successive_elimination.",
    )
    args = parser.parse_args()

    candidates = ["gpt-4o", "gpt-4o-mini"]
    if args.use_instances:
        models = {
            "planner": [ChatOpenAI(model=m) for m in candidates],
            "solver": [ChatOpenAI(model=m) for m in candidates],
        }
    else:
        models = {"planner": candidates, "solver": candidates}

    selector_cls = SELECTORS[args.selector]
    selector_kwargs: Dict[str, Any] = {}
    if args.selector == "random":
        selector_kwargs["sample_fraction"] = args.sample_fraction
    if args.selector in ("hill_climbing", "bayesian_optimization"):
        selector_kwargs["batch_size"] = args.batch_size
    if args.selector == "epsilon_lucb":
        selector_kwargs["epsilon"] = args.epsilon
    if args.selector == "threshold_successive_elimination":
        selector_kwargs["threshold"] = args.threshold

    selector = selector_cls(
        agent_fn=agent_maker,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        **_filter_selector_kwargs(selector_cls, selector_kwargs),
    )

    results = selector.select_best(
        parallel=args.parallel, max_concurrent=args.max_concurrent
    )
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")


if __name__ == "__main__":
    main()
