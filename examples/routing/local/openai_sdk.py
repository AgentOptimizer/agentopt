"""
Example: per-call random routing with an OpenAI Agents SDK agent.

Multi-step Planner+Solver pair.  Each step's underlying LLM call is
routed independently — so a single ``agent.run()`` invocation can hit
two different models, one per step.

Prerequisites:
    1. ``pip install agentopt-py openai-agents python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/openai_sdk.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, function_tool

from agentopt import LLMTracker, RandomRouter


@function_tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"Search results for: {query}"


class MyAgent:
    """OpenAI Agents SDK planner+solver pair with fixed nominal models.

    RandomRouter swaps the model per LLM call; both Planner and Solver
    are routed independently inside the with-block.
    """

    def __init__(self) -> None:
        planner = Agent(
            name="Planner",
            model="gpt-4o-mini",
            instructions="Create a brief plan to answer the question. Be concise.",
            tools=[search],
        )
        solver = Agent(
            name="Solver",
            model="gpt-4o-mini",
            instructions="Given a plan, produce a concise final answer.",
            handoffs=[],
        )
        self.planner = planner.clone(handoffs=[solver])

    def run(self, input_data: str) -> str:
        return Runner.run_sync(self.planner, input_data).final_output


QUESTIONS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "What color is the sky on a clear day?",
    "Name one planet in our solar system.",
    "What is H2O commonly known as?",
]


if __name__ == "__main__":
    agent = MyAgent()
    router = RandomRouter(candidates=["gpt-4o-mini", "gpt-4.1-nano"], seed=0)
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                print(agent.run(q))
    tracker.print_summary()
