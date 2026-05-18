"""
Example: routing for a Gemini CLI subprocess agent.

The ``gemini`` CLI runs as a subprocess and routes its LLM traffic
through agentopt's proxy.  While inside ``track()``, agentopt's
subprocess patch injects ``HTTPS_PROXY`` + the CA bundle into every
spawned child, so the agent's ``run()`` stays free of agentopt imports.
The router runs at the mitmproxy hop just like for in-process Python.

**v1 LIMITATION:** Gemini encodes the model in the URL path
(``/v1beta/models/{model}:generateContent``), not in the request body.
The router is *called* but its decision is logged at DEBUG and the
request passes unrouted.  See the "v1 limits" section of
``docs/api/router.md`` — path-rewrite routing is on the roadmap.

This example is included for shape parity with the selection examples;
once Gemini path-rewrite lands, swapping in the same Python code will
just work.

Prerequisites:
    1. ``pip install agentopt-py``
    2. Install Gemini CLI (https://github.com/google-gemini/gemini-cli)
       and authenticate (``gemini auth`` or ``GEMINI_API_KEY``).

Usage:
    python examples/routing/local/gemini_cli.py
"""

from __future__ import annotations

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

from agentopt import LLMTracker, RandomRouter

GEMINI_TIMEOUT = 120


class MyAgent:
    """Wraps the ``gemini`` CLI as a subprocess."""

    def __init__(self) -> None:
        # The nominal model — overridden by the router (when supported).
        self.model = "gemini-2.5-flash"

    def run(self, prompt: str) -> str:
        cmd = ["gemini", "-m", self.model, "-p", prompt]
        # GEMINI_CLI_TRUST_WORKSPACE=true skips the trusted-folder gate.
        # Agentopt's subprocess patch merges HTTPS_PROXY + CA bundle on top.
        env = {**os.environ, "GEMINI_CLI_TRUST_WORKSPACE": "true"}
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=GEMINI_TIMEOUT, env=env,
            )
        except subprocess.TimeoutExpired:
            return f"FAILED: gemini timeout after {GEMINI_TIMEOUT}s"
        except FileNotFoundError:
            return "FAILED: gemini CLI not found"
        if result.returncode != 0:
            return f"FAILED: gemini exit {result.returncode}: {result.stderr.strip()[:300]}"
        return result.stdout.strip()


QUESTIONS = [
    "What is the capital of France? Answer in one word.",
    "What is 2 + 2? Answer with just the number.",
    "What color is the sky on a clear day? Answer in one word.",
]


if __name__ == "__main__":
    try:
        subprocess.run(["gemini", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Error: gemini CLI not found.")
        sys.exit(1)

    agent = MyAgent()
    # Once Gemini path-rewrite routing lands, this swaps the model per call.
    # For v1, calls pass through unrouted but tracking still works.
    router = RandomRouter(candidates=["gemini-2.5-flash", "gemini-2.5-pro"], seed=0)
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                print(agent.run(q))
    tracker.print_summary()
