"""
Example: AG2 agent with agentopt.

Prerequisites:
    1. pip install ag2 agentopt
    2. Set OPENAI_API_KEY environment variable

Note: AG2's run() API spawns a background thread which breaks context
propagation. This example uses initiate_chat() which blocks in the
caller's thread and is required for agentopt.proxy compatibility.
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import inspect
from typing import Any, Dict

import autogen

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


class AG2Agent:
    """AG2 planner+solver agent pair."""

    def __init__(self, models: Dict[str, Any]):
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

        self.planner = autogen.AssistantAgent(
            name="Planner",
            system_message="You are a planning assistant. Create a brief plan to answer the question. End your response with PLAN_DONE.",
            llm_config=planner_config,
        )
        self.solver = autogen.AssistantAgent(
            name="Solver",
            system_message="You are a solver. Given a plan, produce a concise final answer. End your response with TERMINATE.",
            llm_config=solver_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=4,
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
            code_execution_config=False,
        )

    def run(self, input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        # NOTE: use initiate_chat(), not run().
        # run() spawns a background thread that breaks contextvar propagation.
        # initiate_chat() blocks in the caller's thread — required for agentopt.proxy.
        chat_result = self.user_proxy.initiate_chat(
            self.planner, message=f"Answer this question: {question}", max_turns=4,
        )

        # Extract the last non-empty assistant message as the answer
        for msg in reversed(chat_result.chat_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"].replace("TERMINATE", "").strip()
        return ""


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
    parser = argparse.ArgumentParser(description="AG2 model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument(
        "--use-instances",
        action="store_true",
        help="Pass config dicts instead of model name strings",
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
            "planner": [{"model": m} for m in candidates],
            "solver": [{"model": m} for m in candidates],
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
        agent=AG2Agent,
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
