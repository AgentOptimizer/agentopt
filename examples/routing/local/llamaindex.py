"""
Example: per-call random routing with a LlamaIndex agent.

The LlamaIndex tool-using agent is unchanged; ``with LLMTracker(router=...)``
overrides the model per LLM call.  Works for async run() too.

Prerequisites:
    1. ``pip install agentopt-py llama-index-core llama-index-llms-openai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/llamaindex.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from agentopt import LLMTracker, RandomRouter


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


def divide(a: float, b: float) -> float:
    """Divide a by b and return the result."""
    if b == 0:
        return float("inf")
    return a / b


class MyAgent:
    """LlamaIndex math agent. Nominal model is overridden per call."""

    def __init__(self) -> None:
        llm = LlamaOpenAI(model="gpt-4o-mini")
        agent = FunctionAgent(
            name="MathAgent",
            description="Solves math problems using calculator tools",
            tools=[multiply, add, subtract, divide],
            llm=llm,
            system_prompt=(
                "You are a helpful assistant that can perform mathematical operations. "
                "Use tools when computing."
            ),
        )
        self._workflow = AgentWorkflow(agents=[agent], root_agent="MathAgent")

    async def run(self, input_data: str) -> str:
        response = await self._workflow.run(user_msg=input_data)
        return str(response)


QUESTIONS = [
    "What is 2 + 2?",
    "What is 5 * 3?",
    "What is 10 - 4?",
]


async def main() -> None:
    agent = MyAgent()
    router = RandomRouter(candidates=["gpt-4o-mini", "gpt-4.1-nano"], seed=0)
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                print(await agent.run(q))
    tracker.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
