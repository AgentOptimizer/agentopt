"""
Example: routing for a Terminal Bench (``tb`` CLI) subprocess agent.

``tb run`` launches a Docker container per task and makes LLM calls
through agentopt's proxy.  While inside a ``track()`` scope, agentopt's
subprocess patch injects ``HTTPS_PROXY`` + the CA bundle into every
spawned child, so ``tb`` is intercepted with no env-var plumbing here.
The router runs at the mitmproxy hop and rewrites the model in the
request body.

Note: Docker must be running for ``tb run`` to work at all.

Prerequisites:
    1. ``pip install agentopt-py``
    2. ``uv tool install terminal-bench``   (requires Docker and git)
    3. Docker daemon running.
    4. Set ``OPENAI_API_KEY`` (or ``ANTHROPIC_API_KEY``) per LiteLLM docs.

Usage:
    python examples/routing/local/terminal_bench.py
"""

from __future__ import annotations

import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

from agentopt import LLMTracker, RandomRouter

TB_DATASET = "terminal-bench-core==0.1.1"
TB_AGENT = "terminus"
TB_TIMEOUT = 600


class MyAgent:
    """Wraps ``tb run`` as a subprocess."""

    def __init__(self, model: str = "openai/gpt-4o-mini") -> None:
        # Nominal model — router will override per call.
        self.model = model

    def run(self, task_id: str) -> str:
        cmd = [
            "tb",
            "run",
            "--dataset",
            TB_DATASET,
            "--agent",
            TB_AGENT,
            "--model",
            self.model,
            "--task-id",
            task_id,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=TB_TIMEOUT,
            )
            return result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            return "FAILED: timeout"
        except FileNotFoundError:
            return "FAILED: tb command not found"


TASKS = [
    "hello-world",
    # Add more task_ids; discover with: tb tasks list --dataset terminal-bench-core==head
]


if __name__ == "__main__":
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        if result.returncode != 0:
            print("Error: Docker daemon not running.")
            sys.exit(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Error: docker not available.")
        sys.exit(1)

    agent = MyAgent()
    # Router swaps the model on every LLM call made *inside* each `tb run`.
    router = RandomRouter(candidates=["openai/gpt-4o-mini", "openai/gpt-4o"], seed=0)
    with LLMTracker(router=router) as tracker:
        for t in TASKS:
            with tracker.track(data_id=t):
                print(agent.run(t))
    tracker.print_summary()
