"""
Example: client-side router overrides the daemon's default for one session.

The daemon has a baseline routing policy; the script opens its own
``LLMTracker`` session with a different ``router=`` — that override
wins for the session without changing the daemon's defaults for
everyone else.

Prerequisites:
    1. ``pip install agentopt-py openai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Run in two terminals:

    # Terminal 1 — daemon with a baseline (mini-only) policy
    agentopt serve \\
        --routing-policy random \\
        --candidate-models gpt-4o-mini \\
        --seed 0

    # Terminal 2 — this client overrides with a 2-model pool
    AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000 \\
        python examples/routing/daemon/per_session_override.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from agentopt import LLMTracker, RandomRouter


class MyAgent:
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
    if not os.environ.get("AGENTOPT_GATEWAY_URL"):
        raise SystemExit("Set AGENTOPT_GATEWAY_URL and start `agentopt serve` first.")

    agent = MyAgent()
    # Override the daemon's default for *this* session only.
    router = RandomRouter(candidates=["gpt-4o-mini", "gpt-4.1-nano"], seed=0)
    with LLMTracker(combo_id="override", router=router) as tracker:
        for q in QUESTIONS:
            answer, model = agent.run(q)
            print(f"[{model}] {answer}")
    tracker.print_summary()
