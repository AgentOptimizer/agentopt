"""
Example: per-call random routing with an AG2 agent pair.

Planner+solver agent pair from AG2 (autogen).  ``with LLMTracker(router=...)``
overrides the model per LLM call inside the chat.

Note: use ``initiate_chat()`` not ``run()``.  AG2's ``run()`` spawns a
background thread that breaks ``contextvars`` propagation; ``initiate_chat()``
blocks in the caller's thread so the router is visible.

Prerequisites:
    1. ``pip install agentopt-py ag2 python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/ag2.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import autogen

from agentopt import LLMTracker, RandomRouter


class MyAgent:
    """AG2 planner+solver pair. Nominal model is overridden per call."""

    def __init__(self) -> None:
        self.planner = autogen.AssistantAgent(
            name="Planner",
            system_message=(
                "You are a planning assistant. Create a brief plan to answer the question. "
                "End your response with PLAN_DONE."
            ),
            llm_config={"model": "gpt-4o-mini"},
        )
        self.solver = autogen.AssistantAgent(
            name="Solver",
            system_message=(
                "You are a solver. Given a plan, produce a concise final answer. "
                "End your response with TERMINATE."
            ),
            llm_config={"model": "gpt-4o-mini"},
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=4,
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
            code_execution_config=False,
        )

    def run(self, input_data: str) -> str:
        chat_result = self.user_proxy.initiate_chat(
            self.planner, message=f"Answer this question: {input_data}", max_turns=4,
        )
        for msg in reversed(chat_result.chat_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"].replace("TERMINATE", "").strip()
        return ""


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
