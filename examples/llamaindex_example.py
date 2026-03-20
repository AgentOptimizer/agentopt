"""
Example: LlamaIndex agent with agentopt.

Prerequisites:
    1. pip install llama-index-core llama-index-llms-openai agentopt agentproxy
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import inspect
from typing import Any, Dict

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from agentopt import (
    ArmEliminationModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    LMProposalModelSelector,
    RandomSearchModelSelector,
)

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "lm_proposal": LMProposalModelSelector,
}

try:
    from agentopt import BayesianOptimizationModelSelector

    SELECTORS["bayesian_optimization"] = BayesianOptimizationModelSelector
except ImportError:
    pass


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


class _AsyncRunner:
    """Callable wrapper whose async __call__ is detected by the framework."""

    def __init__(self, workflow):
        self._workflow = workflow

    async def __call__(self, input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]
        response = await self._workflow.run(user_msg=question)
        return str(response)


def agent_maker(models: Dict[str, Any]):
    """Factory: builds a LlamaIndex math agent with the given model."""
    llm = (
        models["agent"]
        if not isinstance(models["agent"], str)
        else LlamaOpenAI(model=models["agent"])
    )

    agent = FunctionAgent(
        name="MathAgent",
        description="Solves math problems using calculator tools",
        tools=[multiply, add, subtract, divide],
        llm=llm,
        system_prompt=(
            "You are a helpful assistant that can perform mathematical operations. "
            "When asked to calculate something, use the available tools to compute the result."
        ),
    )

    workflow = AgentWorkflow(agents=[agent], root_agent="MathAgent")

    return _AsyncRunner(workflow)


def eval_fn(expected: str, actual) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is 2 + 2?", "4"),
    ("What is 5 * 3?", "15"),
    ("What is 10 - 4?", "6"),
]


def _filter_selector_kwargs(
    selector_cls, selector_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    params = inspect.signature(selector_cls.__init__).parameters
    return {k: v for k, v in selector_kwargs.items() if k in params}


def main():
    parser = argparse.ArgumentParser(description="LlamaIndex model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument(
        "--use-instances",
        action="store_true",
        help="Pass pre-built LlamaOpenAI instances instead of model name strings",
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
    args = parser.parse_args()

    candidates = ["gpt-4o", "gpt-4o-mini"]
    if args.use_instances:
        models = {"agent": [LlamaOpenAI(model=m) for m in candidates]}
    else:
        models = {"agent": candidates}

    selector_cls = SELECTORS[args.selector]
    selector_kwargs: Dict[str, Any] = {}
    if args.selector == "random":
        selector_kwargs["sample_fraction"] = args.sample_fraction
    if args.selector in ("hill_climbing", "bayesian_optimization"):
        selector_kwargs["batch_size"] = args.batch_size

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
