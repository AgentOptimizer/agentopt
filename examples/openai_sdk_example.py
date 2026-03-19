"""
Example: OpenAI Agents SDK agent with agentopt.

Prerequisites:
    1. pip install openai-agents agentopt agentproxy
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
from typing import Any, Dict

from agents import Agent, Runner, function_tool

from agentopt import (
    ArmEliminationModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    HyperbandModelSelector,
    LMProposalModelSelector,
    RandomSearchModelSelector,
)

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "hyperband": HyperbandModelSelector,
    "lm_proposal": LMProposalModelSelector,
}

try:
    from agentopt import BayesianOptimizationModelSelector

    SELECTORS["bayesian_optimization"] = BayesianOptimizationModelSelector
except ImportError:
    pass


@function_tool
def search(query: str) -> str:
    """Search for information about a topic."""
    # Stub — replace with a real search tool as needed.
    return f"Search results for: {query}"


def _resolve_model(candidate: Any) -> str:
    """Extract model name string from a candidate (string, dict, or object)."""
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return candidate["model"]
    return getattr(candidate, "model", str(candidate))


def agent_maker(models: Dict[str, Any]):
    """Factory: builds an OpenAI Agents SDK planner+solver agent pair."""
    planner = Agent(
        name="Planner",
        model=_resolve_model(models["planner"]),
        instructions="Create a brief plan to answer the question. Be concise.",
        tools=[search],
    )
    solver = Agent(
        name="Solver",
        model=_resolve_model(models["solver"]),
        instructions="Given a plan, produce a concise final answer.",
        handoffs=[],
    )

    # Wire planner → solver via handoff
    planner_with_handoff = planner.clone(handoffs=[solver])

    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]
        result = Runner.run_sync(planner_with_handoff, question)
        return result.final_output

    return run


def eval_fn(expected: str, actual) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
]


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Agents SDK model selection example"
    )
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument(
        "--use-instances",
        action="store_true",
        help="Pass config dicts instead of model name strings",
    )
    args = parser.parse_args()

    candidates = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
    if args.use_instances:
        models = {
            "planner": [{"model": m} for m in candidates],
            "solver": [{"model": m} for m in candidates],
        }
    else:
        models = {"planner": candidates, "solver": candidates}

    selector_cls = SELECTORS[args.selector]
    selector = selector_cls(
        agent_fn=agent_maker, models=models, eval_fn=eval_fn, dataset=dataset,
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
