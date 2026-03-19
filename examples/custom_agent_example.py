"""
Example: Custom agent (no framework) with agentopt.

This example shows how to use agentopt with a plain Python agent
that makes OpenAI SDK calls directly. No framework or proxy needed.

Prerequisites:
    1. pip install openai agentopt agentproxy
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
from typing import Any, Dict

from openai import OpenAI

from agentopt import (
    ArmEliminationModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    HyperbandModelSelector,
    LMProposalModelSelector,
    RandomSearchModelSelector,
)
from agentproxy import LLMTracker, ResponseCache

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


def _resolve_model(candidate: Any) -> str:
    """Extract model name string from a candidate (string, dict, or object)."""
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return candidate["model"]
    return getattr(candidate, "model", str(candidate))


def agent_maker(models: Dict[str, Any]):
    """Factory: builds a simple planner+solver agent for the given models."""
    client = OpenAI()

    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        # Step 1: Planner generates a plan
        plan_response = client.chat.completions.create(
            model=_resolve_model(models["planner"]),
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
            model=_resolve_model(models["solver"]),
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
    tracker = LLMTracker(cache_dir="./llm_cache")
    selector = selector_cls(
        agent_fn=agent_maker,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        tracker=tracker,
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

    # Show cache DB on disk
    from pathlib import Path

    db_path = Path("./llm_cache/cache.db")
    if db_path.exists():
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        (count,) = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        conn.close()
        print(f"\nCache: {count} entries saved to {db_path}")


if __name__ == "__main__":
    main()
