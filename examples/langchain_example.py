"""
Example: LangChain agent with agentopt.

Prerequisites:
    1. pip install langchain langchain-openai agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import inspect
from typing import Any, Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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


@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    # Stub — replace with a real search tool as needed.
    return f"Search results for: {query}"


TOOLS = [search]

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use tools when needed to answer questions concisely.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def agent_maker(models: Dict[str, Any]):
    """Factory: builds a LangChain tool-calling agent."""
    llm = (
        models["agent"]
        if not isinstance(models["agent"], str)
        else ChatOpenAI(model=models["agent"], disable_streaming=True)
    )

    agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
    executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)

    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]
        result = executor.invoke({"input": question})
        return result["output"]

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


def _filter_selector_kwargs(
    selector_cls, selector_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    params = inspect.signature(selector_cls.__init__).parameters
    return {k: v for k, v in selector_kwargs.items() if k in params}


def main():
    parser = argparse.ArgumentParser(description="LangChain model selection example")
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

    candidates = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
    if args.use_instances:
        models = {"agent": [ChatOpenAI(model=m) for m in candidates]}
    else:
        models = {"agent": candidates}

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
    results.plot_pareto()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")


if __name__ == "__main__":
    main()
