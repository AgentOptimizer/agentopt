"""
Example: per-call random routing with a LangChain tool-calling agent.

Same clean API as the plain-Python example — the LangChain agent is
unmodified; only the ``with LLMTracker(router=...)`` wrapper differs.

Prerequisites:
    1. ``pip install agentopt-py langchain-classic langchain-openai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/langchain.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agentopt import LLMTracker, RandomRouter


@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"Search results for: {query}"


TOOLS = [search]

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use tools when needed; answer concisely.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


class MyAgent:
    """LangChain tool-calling agent with a fixed nominal model.

    RandomRouter swaps the model per LLM call inside the with-block; the
    agent itself doesn't know anything about routing.
    """

    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-4o-mini", disable_streaming=True)
        agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
        self.executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)

    def run(self, question: str) -> str:
        return self.executor.invoke({"input": question})["output"]


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
