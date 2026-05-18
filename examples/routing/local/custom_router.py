"""
Example: writing your own Router policy.

``LengthBasedRouter`` inspects each request's prompt and sends long
prompts to a stronger model, short ones to a cheaper one.  This is the
typical routing pattern — pick by request content.

``ctx.request_body`` is a read-only view of the parsed inbound JSON;
``ctx.history`` would let you decide based on prior calls.

Prerequisites:
    1. ``pip install agentopt-py openai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/custom_router.py
"""

from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from agentopt import LLMTracker, Router
from agentopt.routing import RouteContext


class LengthBasedRouter(Router):
    """Long prompts → big model, short prompts → small model."""

    def __init__(self, big: str, small: str, threshold_chars: int = 120) -> None:
        self.big = big
        self.small = small
        self.threshold_chars = threshold_chars

    def route(self, ctx: RouteContext) -> Optional[str]:
        messages = ctx.request_body.get("messages") or []
        total = sum(len(str(m.get("content", ""))) for m in messages)
        return self.big if total > self.threshold_chars else self.small


class MyAgent:
    def __init__(self) -> None:
        self.client = OpenAI()

    def run(self, question: str):
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": question},
            ],
        )
        return resp.choices[0].message.content, resp.model


QUESTIONS = [
    "2+2?",  # short → small
    "Sky color?",  # short → small
    (
        "Compare quicksort and mergesort in terms of average-case time complexity, "
        "stability, and typical memory overhead.  Give one concrete scenario where "
        "you would prefer each."
    ),  # long → big
    "Capital of France?",  # short → small
]


if __name__ == "__main__":
    agent = MyAgent()
    router = LengthBasedRouter(big="gpt-4o", small="gpt-4o-mini")
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                answer, model = agent.run(q)
                print(f"[{model}] Q={q[:50]!r}{'...' if len(q) > 50 else ''}")
                print(f"  A: {answer}\n")
    tracker.print_summary()
