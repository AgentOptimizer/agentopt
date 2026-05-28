"""
Example: per-call random routing with a plain-Python OpenAI agent.

The whole user-facing routing API in one shot:

    router = RandomRouter(candidates=[...], seed=0)
    with LLMTracker(combo_id="demo", router=router) as tracker:
        for dp in datapoints:
            agent.run(dp)
    tracker.print_summary()

Inside the ``with`` block, every LLM call is independently routed to one
of the candidates regardless of what the SDK passes for ``model=``.

To verify the swap visibly: each agent call returns ``(answer, resp.model)``
— ``resp.model`` is the actual model the API used, echoed by the server.

Prerequisites:
    1. ``pip install agentopt-py openai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/custom_agent.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from agentopt import LLMTracker, RandomRouter


class MyAgent:
    """One-shot OpenAI chat completion.

    The ``model=`` it passes is the SDK's *request* — RandomRouter will
    override it on every call.
    """

    def __init__(self) -> None:
        self.client = OpenAI()

    def run(self, question: str):
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": question},
            ],
        )
        return resp.choices[0].message.content, resp.model


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
    # Open a per-question session so the routing summary groups by datapoint.
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                answer, model = agent.run(q)
                print(f"[{model}] {answer}")
    tracker.print_summary()
