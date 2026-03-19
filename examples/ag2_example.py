"""
Example: AG2 agent with agentopt.

Prerequisites:
    1. pip install ag2 agentopt agentproxy
    2. Set OPENAI_API_KEY environment variable

Note: AG2's run() API spawns a background thread which breaks context
propagation. This example uses initiate_chat() which blocks in the
caller's thread and is required for agentproxy compatibility.
"""

import argparse
from typing import Any, Dict

import autogen

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


def agent_maker(models: Dict[str, Any]):
    """Factory: builds an AG2 planner+solver agent pair."""
    planner_config = (
        models["planner"]
        if isinstance(models["planner"], dict)
        else {"model": models["planner"]}
    )
    solver_config = (
        models["solver"]
        if isinstance(models["solver"], dict)
        else {"model": models["solver"]}
    )

    planner = autogen.AssistantAgent(
        name="Planner",
        system_message="You are a planning assistant. Create a brief plan to answer the question. End your response with PLAN_DONE.",
        llm_config=planner_config,
    )
    solver = autogen.AssistantAgent(
        name="Solver",
        system_message="You are a solver. Given a plan, produce a concise final answer. End your response with TERMINATE.",
        llm_config=solver_config,
    )
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=4,
        is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
        code_execution_config=False,
    )

    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        # NOTE: use initiate_chat(), not run().
        # run() spawns a background thread that breaks contextvar propagation.
        # initiate_chat() blocks in the caller's thread — required for agentproxy.
        chat_result = user_proxy.initiate_chat(
            planner, message=f"Answer this question: {question}", max_turns=4,
        )

        # Extract the last non-empty assistant message as the answer
        for msg in reversed(chat_result.chat_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"].replace("TERMINATE", "").strip()
        return ""

    return run


def eval_fn(expected: str, actual) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
]


def main():
    parser = argparse.ArgumentParser(description="AG2 model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument(
        "--use-instances",
        action="store_true",
        help="Pass config dicts instead of model name strings",
    )
    args = parser.parse_args()

    candidates = ["gpt-4o", "gpt-4o-mini"]
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
