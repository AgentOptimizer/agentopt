"""
Example: daemon-wide default routing — client code knows nothing about it.

The daemon is started with a default routing policy via CLI flags.
The Python client just opens a tracking session — no router code — and
the daemon applies its default policy to every LLM call.

This is the shared-infrastructure pattern: one daemon's policy applies
to every team script transparently.  The ``with LLMTracker(combo_id=...)``
is what plugs the Python client's httpx into the daemon for this script;
without it, calls would go straight to ``api.openai.com`` and bypass the
daemon entirely.

Prerequisites:
    1. ``pip install agentopt-py openai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Run in two terminals:

    # Terminal 1 — daemon with a default RandomRouter
    agentopt serve \\
        --routing-policy random \\
        --candidate-models gpt-4o-mini,gpt-4.1-nano \\
        --seed 0

    # Terminal 2 — client opens a session; daemon does the routing
    AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000 \\
        python examples/routing/daemon/default_policy.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from agentopt import LLMTracker


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
        raise SystemExit(
            "Set AGENTOPT_GATEWAY_URL and start `agentopt serve "
            "--routing-policy random --candidate-models gpt-4o-mini,gpt-4.1-nano` first."
        )

    agent = MyAgent()
    # No router= here — the daemon applies its --routing-policy default.
    with LLMTracker(combo_id="daemon-default") as tracker:
        for q in QUESTIONS:
            answer, model = agent.run(q)
            print(f"[{model}] {answer}")
    tracker.print_summary()
