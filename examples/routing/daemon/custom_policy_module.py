"""
Example: custom Router run through the daemon.

The router class lives in ``my_policies.py`` next to this file.  The
daemon imports it via ``--policy-module``; the client imports the same
file so the wire protocol round-trips (``module:Class`` resolves on
both ends).

Prerequisites:
    1. ``pip install agentopt-py openai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Run in two terminals:

    # Terminal 1 — daemon pre-imports the policy module
    agentopt serve \\
        --policy-module examples/routing/daemon/my_policies.py

    # Terminal 2 — client imports the same file by path
    AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000 \\
        python examples/routing/daemon/custom_policy_module.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make ``my_policies`` importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI  # noqa: E402

from agentopt import LLMTracker  # noqa: E402
from my_policies import LengthBasedRouter  # noqa: E402


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
    "2+2?",
    "Sky color?",
    (
        "Compare quicksort and mergesort in terms of average-case time complexity, "
        "stability, and typical memory overhead.  Give one concrete scenario where "
        "you would prefer each."
    ),
    "Capital of France?",
]


if __name__ == "__main__":
    if not os.environ.get("AGENTOPT_GATEWAY_URL"):
        raise SystemExit(
            "Set AGENTOPT_GATEWAY_URL and start `agentopt serve "
            "--policy-module examples/routing/daemon/my_policies.py` first."
        )

    agent = MyAgent()
    router = LengthBasedRouter(big="gpt-4o", small="gpt-4o-mini")
    with LLMTracker(combo_id="custom", router=router) as tracker:
        for q in QUESTIONS:
            answer, model = agent.run(q)
            print(f"[{model}] Q={q[:50]!r}{'...' if len(q) > 50 else ''}")
            print(f"  A: {answer}\n")
    tracker.print_summary()
