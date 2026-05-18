"""
Example: per-call random routing with a LangGraph agent.

The LangGraph planner+solver graph is unchanged from the selection
example; the only difference is the ``with LLMTracker(router=...)`` wrapper.
Each step's LLM call inside ``agent.run(...)`` is routed independently.

Prerequisites:
    1. ``pip install agentopt-py langchain-openai langgraph python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/langgraph.py
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agentopt import LLMTracker, RandomRouter


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    answer: str


class MyAgent:
    """LangGraph planner+solver graph. Nominal model is overridden per call."""

    def __init__(self) -> None:
        planner_llm = ChatOpenAI(model="gpt-4o-mini")
        solver_llm = ChatOpenAI(model="gpt-4o-mini")

        def planner_node(state: AgentState) -> dict:
            response = planner_llm.invoke(
                [{"role": "system", "content": "Plan briefly."}] + state["messages"]
            )
            return {"plan": response.content}

        def solver_node(state: AgentState) -> dict:
            response = solver_llm.invoke(
                [
                    {
                        "role": "system",
                        "content": f"Follow plan, answer concisely:\n{state['plan']}",
                    },
                    state["messages"][-1],
                ]
            )
            return {"answer": response.content}

        graph = StateGraph(AgentState)
        graph.add_node("planner", planner_node)
        graph.add_node("solver", solver_node)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "solver")
        graph.add_edge("solver", END)
        self._app = graph.compile()

    def run(self, input_data: str) -> str:
        result = self._app.invoke(
            {"messages": [{"role": "user", "content": input_data}]}
        )
        return result["answer"]


QUESTIONS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "What color is the sky on a clear day?",
]


if __name__ == "__main__":
    agent = MyAgent()
    router = RandomRouter(candidates=["gpt-4o-mini", "gpt-4.1-nano"], seed=0)
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                print(agent.run(q))
    tracker.print_summary()
