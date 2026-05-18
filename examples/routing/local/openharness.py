"""
Example: routing for an OpenHarness (``oh`` CLI) subprocess agent.

``oh`` calls the official Anthropic/OpenAI SDKs under the hood, which
honour ``HTTPS_PROXY`` set by ``agentopt.get_current_session_proxy()``.
The router runs at the mitmproxy hop and rewrites the model in the
request body.

Prerequisites:
    1. ``pip install agentopt-py``
    2. ``pip install openharness-ai`` (or ``uv tool install openharness-ai``)
    3. Set ``OPENAI_API_KEY`` (or ``ANTHROPIC_API_KEY`` for the anthropic format).

Usage:
    python examples/routing/local/openharness.py
"""

from __future__ import annotations

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

import agentopt
from agentopt import LLMTracker, RandomRouter

OH_TIMEOUT = 120

PROVIDER_DEFAULTS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


class MyAgent:
    """Wraps the ``oh`` CLI as a subprocess."""

    def __init__(self, api_format: str = "openai", model: str = "gpt-4o-mini") -> None:
        if api_format not in PROVIDER_DEFAULTS:
            raise ValueError(f"unsupported api_format {api_format!r}")
        self.api_format = api_format
        # Nominal model — router will override per call.
        self.model = model
        defaults = PROVIDER_DEFAULTS[api_format]
        api_key = os.environ.get(defaults["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{defaults['api_key_env']} not set")
        self.api_key = api_key
        self.base_url = defaults["base_url"]

    def run(self, prompt: str) -> str:
        cmd = [
            "oh",
            "-p",
            prompt,
            "--api-format",
            self.api_format,
            "--base-url",
            self.base_url,
            "--api-key",
            self.api_key,
            "--model",
            self.model,
            "--bare",
            "--dangerously-skip-permissions",
            "--max-turns",
            "1",
            "--output-format",
            "text",
        ]
        proxy = agentopt.get_current_session_proxy()
        env = {**os.environ, **(proxy.env_dict() if proxy else {})}
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=OH_TIMEOUT, env=env,
            )
        except subprocess.TimeoutExpired:
            return f"FAILED: oh timeout after {OH_TIMEOUT}s"
        except FileNotFoundError:
            return "FAILED: oh CLI not found"
        if result.returncode != 0:
            return f"FAILED: oh exit {result.returncode}: {result.stderr.strip()[:300]}"
        return result.stdout.strip()


QUESTIONS = [
    "What is the capital of France? Answer in one word.",
    "What is 2 + 2? Answer with just the number.",
    "What color is the sky on a clear day? Answer in one word.",
]


if __name__ == "__main__":
    try:
        subprocess.run(["oh", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Error: `oh` CLI not found. Install with `pip install openharness-ai`.")
        sys.exit(1)

    agent = MyAgent(api_format="openai")
    router = RandomRouter(candidates=["gpt-4o-mini", "gpt-4.1-nano"], seed=0)
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                print(agent.run(q))
    tracker.print_summary()
