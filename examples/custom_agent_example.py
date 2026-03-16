"""
Example: Custom agent (no framework) with agentopt.

This example shows how to use agentopt with a plain Python agent
that makes OpenAI SDK calls directly. No framework or proxy needed.

Prerequisites:
    1. pip install openai agentopt agentproxy
    2. Set OPENAI_API_KEY environment variable
"""

import argparse
from typing import Dict

from openai import OpenAI

from agentopt import (
    ArmEliminationModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    HyperbandModelSelector,
    RandomSearchModelSelector,
)

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "hyperband": HyperbandModelSelector,
}

try:
    from agentopt import BayesianOptimizationModelSelector

    SELECTORS["bayesian_optimization"] = BayesianOptimizationModelSelector
except ImportError:
    pass


def agent_maker(models: Dict[str, str]):
    """Factory: builds a simple planner+solver agent for the given models."""
    client = OpenAI()

    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        # Step 1: Planner generates a plan
        plan_response = client.chat.completions.create(
            model=models["planner"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a planning assistant. Create a brief plan to answer the question.",
                },
                {"role": "user", "content": question},
            ],
        )
        plan = plan_response.choices[0].message.content

        # Step 2: Solver executes the plan
        solve_response = client.chat.completions.create(
            model=models["solver"],
            messages=[
                {
                    "role": "system",
                    "content": f"Follow this plan and answer concisely:\n{plan}",
                },
                {"role": "user", "content": question},
            ],
        )
        return solve_response.choices[0].message.content

    return run


def eval_fn(expected: str, actual) -> float:
    """Simple evaluation: check if the expected answer appears in the actual response."""
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


# Dataset: (input_data, expected_answer) pairs
dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
]


def main():
    parser = argparse.ArgumentParser(description="Custom agent model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    args = parser.parse_args()

    selector_cls = SELECTORS[args.selector]
    selector = selector_cls(
        agent_fn=agent_maker,
        models={
            "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
            "solver": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
        },
        eval_fn=eval_fn,
        dataset=dataset,
    )

    results = selector.select_best(
        parallel=args.parallel, max_concurrent=args.max_concurrent
    )
    results.print_summary()

    # Export optimized config
    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
        results.export_config("litellm_config_optimized.yaml")
        print("Exported optimized config to litellm_config_optimized.yaml")


if __name__ == "__main__":
    main()
