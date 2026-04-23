"""
Example: per-call model routing with LangGraph + agentopt.

A ``Router`` is a policy that decides — per individual LLM call — which
model to actually send the request to.  Usage is a single context manager
around the agent call; the swap happens transparently at the HTTP layer,
so any framework (LangGraph, LangChain, OpenAI SDK, subprocess agents)
works without integration code.

    router = RandomRouter(model_candidates=["gpt-4o", "gpt-4o-mini"])
    with router:
        answer = agent.run(question)

This example runs a 2-step LangGraph agent (planner → solver) under two
routers:

1. ``RandomRouter`` — uniform pick from a pool, per call.
2. A small custom router that uses ``ctx.history`` to send the first LLM
   call of a workflow to a bigger model and subsequent calls to a
   cheaper one.

Prerequisites:
    pip install langchain-openai langgraph agentopt-py python-dotenv
    export OPENAI_API_KEY=...
"""

from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agentopt import RandomRouter, RouteContext, RouteDecision, Router


# ---------------------------------------------------------------------------
# Step 1: A plain LangGraph planner+solver agent.  Nothing here knows
# about routing — the model passed to ``ChatOpenAI`` is the "requested"
# model; any active router replaces it at the HTTP layer.
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    answer: str


class MyAgent:
    def __init__(self, default_model: str = "gpt-4o-mini") -> None:
        planner_llm = ChatOpenAI(model=default_model)
        solver_llm = ChatOpenAI(model=default_model)

        def planner_node(state: AgentState) -> dict:
            resp = planner_llm.invoke(
                [{"role": "system", "content": "Write a one-line plan."}]
                + state["messages"]
            )
            return {"plan": resp.content}

        def solver_node(state: AgentState) -> dict:
            resp = solver_llm.invoke(
                [
                    {
                        "role": "system",
                        "content": f"Follow this plan and answer concisely:\n{state['plan']}",
                    },
                    state["messages"][-1],
                ]
            )
            return {"answer": resp.content}

        graph = StateGraph(AgentState)
        graph.add_node("planner", planner_node)
        graph.add_node("solver", solver_node)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "solver")
        graph.add_edge("solver", END)
        self._app = graph.compile()

    def run(self, question: str) -> str:
        out = self._app.invoke({"messages": [{"role": "user", "content": question}]})
        return out["answer"]


# ---------------------------------------------------------------------------
# Step 2: A custom policy — inherit from ``Router`` to get ``with``
# support for free, and implement ``route`` to return a decision.
# ---------------------------------------------------------------------------


class FirstCallBigRouter(Router):
    """Big model for the first call of a session, cheap model afterwards."""

    def __init__(self, big: str, small: str) -> None:
        self.big = big
        self.small = small

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        # ctx.history is a snapshot of prior LLM calls in the active
        # routing scope — empty on the very first call.
        if len(ctx.history) == 0:
            return RouteDecision(model=self.big)
        return RouteDecision(model=self.small)


# ---------------------------------------------------------------------------
# Step 3: Run the agent under each router.
# ---------------------------------------------------------------------------


questions = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "What color is the sky on a clear day?",
]


def demo(label: str, router: Router) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    agent = MyAgent(default_model="gpt-4o-mini")

    for q in questions:
        with router:
            answer = agent.run(q)
        print(f"\nQ: {q}\nA: {answer}")


if __name__ == "__main__":
    # Demo A — uniform random pick from a pool of two models.
    demo(
        "Demo A: RandomRouter(['gpt-4o', 'gpt-4o-mini'])",
        RandomRouter(model_candidates=["gpt-4o", "gpt-4o-mini"], seed=0),
    )

    # Demo B — big-first, cheap-rest policy driven by ctx.history.
    demo(
        "Demo B: FirstCallBigRouter(big='gpt-4o', small='gpt-4o-mini')",
        FirstCallBigRouter(big="gpt-4o", small="gpt-4o-mini"),
    )
