"""
Example: same selection script, local proxy vs. ``agentopt serve`` daemon.

The user-facing API does not change between local and daemon modes; the
``AGENTOPT_GATEWAY_URL`` env var is the entire switch.

Local mode (default — spins up an in-process mitmproxy per session)::

    python examples/selection/daemon/basic.py

Daemon mode (one long-lived gateway shared by every script and language)::

    # Terminal 1 — start the daemon (Ctrl-C to stop).
    agentopt serve

    # Terminal 2 — same script, just point at the daemon.
    AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000 \\
        python examples/selection/daemon/basic.py

The selector script (below) is byte-identical between the two runs.

Prerequisites:
    1. ``pip install agentopt-py openai``
    2. ``export OPENAI_API_KEY=...``
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from agentopt import ModelSelector


# ---------------------------------------------------------------------------
# Step 1: Define your agent class.
# ---------------------------------------------------------------------------


class MyAgent:
    """One-shot OpenAI chat completion."""

    def __init__(self, models):
        self.client = OpenAI()
        self.model = models["agent"]

    def run(self, input_data: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer in one word."},
                {"role": "user", "content": input_data},
            ],
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Step 2: Dataset + scoring.
# ---------------------------------------------------------------------------


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
]


def eval_fn(expected: str, actual: str) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


# ---------------------------------------------------------------------------
# Step 3: Run.  Nothing here knows about local vs. daemon mode.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    gateway = os.environ.get("AGENTOPT_GATEWAY_URL")
    print(f"Mode: {'remote → ' + gateway if gateway else 'local (in-process proxy)'}")

    selector = ModelSelector(
        agent=MyAgent,
        models={"agent": ["gpt-4o-mini", "gpt-4.1-nano"]},
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",
    )
    results = selector.select_best(parallel=False)
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest model: {best}")
