"""
Example: LangChain agent with agentopt.

Prerequisites:
    1. pip install langchain langchain-openai agentopt agentproxy
    2. Set OPENAI_API_KEY environment variable
"""

import argparse
from typing import Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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


def agent_maker(models: Dict[str, str]):
    """Factory: builds a LangChain tool-calling agent."""
    llm = ChatOpenAI(model=models["agent"])

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


def main():
    parser = argparse.ArgumentParser(description="LangChain model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    args = parser.parse_args()

    selector_cls = SELECTORS[args.selector]
    selector = selector_cls(
        agent_fn=agent_maker,
        models={"agent": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],},
        eval_fn=eval_fn,
        dataset=dataset,
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
