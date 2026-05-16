"""
Example: OpenHarness (``oh`` CLI) with agentopt.

Find the best LLM model for your OpenHarness tasks. The agent wraps the
``oh`` CLI as a subprocess; agentopt sets ``HTTPS_PROXY`` and a CA bundle
so all LLM traffic from the subprocess routes through the tracking proxy,
and records model/tokens/latency per call.

The ``oh`` CLI (https://github.com/HKUDS/OpenHarness) is a Python tool
that uses the official Anthropic/OpenAI SDKs under the hood, which means
it honours ``HTTPS_PROXY`` / ``SSL_CERT_FILE`` env vars out of the box —
no config-file patching needed. ``--api-key`` and ``--base-url`` flags
let us bypass interactive ``oh setup`` entirely.

Prerequisites:
    1. pip install agentopt-py
    2. Install OpenHarness:  pip install openharness-ai   (or `uv tool install openharness-ai`)
    3. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in your environment.

Usage:
    python examples/openharness_example.py
"""

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

import agentopt
from agentopt import ModelSelector

OH_TIMEOUT = 120  # seconds per call


# ---------------------------------------------------------------------------
# Step 1: Define the agent class.
# __init__(models) receives a model config dict; run(input_data) shells out
# to `oh -p` and returns the stdout text.
#
# Model strings use a "<format>/<model_id>" convention so one agent can
# select across Anthropic and OpenAI-compatible providers:
#   "anthropic/claude-haiku-4-5"  ->  --api-format anthropic
#   "openai/gpt-4o-mini"          ->  --api-format openai
# ---------------------------------------------------------------------------


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


class OpenHarnessAgent:
    """Wraps the ``oh`` CLI as a subprocess for agentopt model selection."""

    def __init__(self, models):
        spec = models["agent"]
        if "/" not in spec:
            raise ValueError(
                f"model must be '<format>/<model_id>' (e.g. 'anthropic/claude-haiku-4-5'); got {spec!r}"
            )
        self.api_format, self.model = spec.split("/", 1)
        if self.api_format not in PROVIDER_DEFAULTS:
            raise ValueError(
                f"unsupported api_format {self.api_format!r}; expected one of {list(PROVIDER_DEFAULTS)}"
            )
        defaults = PROVIDER_DEFAULTS[self.api_format]
        api_key = os.environ.get(defaults["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{defaults['api_key_env']} not set in environment")
        self.api_key = api_key
        self.base_url = defaults["base_url"]

    def run(self, input_data):
        prompt = input_data if isinstance(input_data, str) else input_data["prompt"]

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
            "--bare",  # skip hooks/plugins/MCP/auto-discovery
            "--dangerously-skip-permissions",  # non-interactive, no permission prompts
            "--max-turns",
            "1",  # single turn for eval
            "--output-format",
            "text",
        ]

        # Route subprocess LLM calls through agentopt's tracking proxy.
        # `oh` uses the official anthropic/openai SDKs, which read these env
        # vars via their default httpx clients.
        proxy = agentopt.get_current_session_proxy()
        env = {
            **os.environ,
            **(proxy.env_dict() if proxy else {}),
        }

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=OH_TIMEOUT, env=env,
            )
        except subprocess.TimeoutExpired:
            return f"FAILED: oh timeout after {OH_TIMEOUT}s"
        except FileNotFoundError:
            return "FAILED: oh CLI not found — install with: pip install openharness-ai"

        if result.returncode != 0:
            return f"FAILED: oh exit {result.returncode}: {result.stderr.strip()[:300]}"
        return result.stdout.strip()


# ---------------------------------------------------------------------------
# Step 2: Evaluation dataset — (prompt, expected_answer) pairs.
# ---------------------------------------------------------------------------

dataset = [
    ("What is the capital of France? Answer in one word.", "Paris"),
    ("What is 2 + 2? Answer with just the number.", "4"),
    ("What color is the sky on a clear day? Answer in one word.", "blue"),
]


# ---------------------------------------------------------------------------
# Step 3: Evaluation function — substring match with debug surfacing.
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    actual_str = str(actual)
    if "FAILED" in actual_str:
        print(f"  [debug] sample FAILED: {actual_str[:300]}")
        return 0.0
    matched = expected.lower() in actual_str.lower()
    if not matched:
        print(f"  [debug] no match — expected={expected!r}, got={actual_str[:200]!r}")
    return 1.0 if matched else 0.0


# ---------------------------------------------------------------------------
# Step 4: Run model selection.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        subprocess.run(["oh", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(
            "Error: `oh` CLI not found. Install with `pip install openharness-ai` "
            "or `uv tool install openharness-ai`."
        )
        sys.exit(1)

    selector = ModelSelector(
        agent=OpenHarnessAgent,
        models={"agent": ["openai/gpt-4o-mini", "openai/gpt-5.1", "openai/gpt-4.1",],},
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",
    )

    results = selector.select_best(parallel=False)
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest model: {best}")
