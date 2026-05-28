"""
Example: routing for an OpenClaw subprocess agent.

OpenClaw's gateway daemon ignores ``HTTPS_PROXY``, so the
``OpenClawAgent`` wrapper (defined in ``examples/shared/openclaw_agent.py``
— shared with the selection example) patches ``~/.openclaw/openclaw.json``
with the proxy URL + CA cert before each inference and restores it
after.  The router runs at the mitmproxy hop in the same way as for the
other subprocess examples.

Prerequisites:
    1. ``pip install agentopt-py``
    2. ``npm install -g openclaw``
    3. ``openclaw onboard`` to set up at least one provider.
    4. ``export ANTHROPIC_API_KEY=...``

Usage:
    python examples/routing/local/openclaw.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# OpenClawAgent is shared with examples/selection/local/openclaw.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from agentopt import LLMTracker, RandomRouter  # noqa: E402
from openclaw_agent import OpenClawAgent  # noqa: E402


class MyAgent:
    """Thin wrapper over OpenClawAgent to match the no-args routing pattern."""

    def __init__(self, model: str = "anthropic/claude-haiku-4-5") -> None:
        # OpenClawAgent expects {"agent": <model>} — give it a nominal model;
        # the router overrides per call.
        self._inner = OpenClawAgent(models={"agent": model})

    def run(self, prompt: str) -> str:
        return self._inner.run(prompt)


QUESTIONS = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "What color is the sky on a clear day?",
]


if __name__ == "__main__":
    try:
        subprocess.run(
            ["openclaw", "--version"],
            capture_output=True,
            timeout=10,
            env={
                **os.environ,
                "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}",
            },
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Error: openclaw CLI not found. Install with: npm install -g openclaw")
        sys.exit(1)

    config_path = os.path.expanduser("~/.openclaw/openclaw.json")
    if not os.path.exists(config_path):
        print(
            f"Error: OpenClaw config not found at {config_path}. Run `openclaw onboard`."
        )
        sys.exit(1)

    agent = MyAgent()
    router = RandomRouter(
        candidates=["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6"],
        seed=0,
    )
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                print(agent.run(q))
    tracker.print_summary()
