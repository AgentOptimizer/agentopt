"""
Example: Gemini CLI with agentopt.

Find the best Gemini model for your tasks. The agent wraps the ``gemini``
CLI as a subprocess; while a ``track()`` scope is active, agentopt's
subprocess patch transparently injects ``HTTPS_PROXY`` + the CA bundle
into the child's env so LLM traffic routes through the tracking proxy
and we get model/tokens/latency per call.

Prerequisites:
    1. pip install agentopt-py
    2. Install Gemini CLI (https://github.com/google-gemini/gemini-cli)
       and authenticate it (`gemini auth` or set GEMINI_API_KEY).
    3. Optional: set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.

Usage:
    python examples/selection/local/gemini_cli.py
"""

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

from agentopt import ModelSelector

GEMINI_TIMEOUT = 120  # seconds per call


# ---------------------------------------------------------------------------
# Step 1: Define the agent class.
# __init__(models) receives a model config dict; run(input_data) shells out
# to `gemini` and returns the stdout text.
# ---------------------------------------------------------------------------


class GeminiCLIAgent:
    """Wraps the ``gemini`` CLI as a subprocess for agentopt model selection."""

    def __init__(self, models):
        self.model = models["agent"]

    def run(self, input_data):
        prompt = input_data if isinstance(input_data, str) else input_data["prompt"]
        cmd = ["gemini", "-m", self.model, "-p", prompt]

        # GEMINI_CLI_TRUST_WORKSPACE=true skips Gemini CLI's trusted-folder
        # gate, which otherwise blocks headless execution.  Agentopt's
        # subprocess patch will merge HTTPS_PROXY + CA bundle on top of this.
        env = {**os.environ, "GEMINI_CLI_TRUST_WORKSPACE": "true"}
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=GEMINI_TIMEOUT, env=env,
            )
        except subprocess.TimeoutExpired:
            return f"FAILED: gemini timeout after {GEMINI_TIMEOUT}s"
        except FileNotFoundError:
            return "FAILED: gemini CLI not found — see https://github.com/google-gemini/gemini-cli"

        if result.returncode != 0:
            return f"FAILED: gemini exit {result.returncode}: {result.stderr.strip()[:300]}"
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
    # Verify gemini CLI is installed and on PATH
    try:
        subprocess.run(["gemini", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(
            "Error: gemini CLI not found. "
            "Install from https://github.com/google-gemini/gemini-cli, "
            "then run `gemini auth` or set GEMINI_API_KEY."
        )
        sys.exit(1)

    selector = ModelSelector(
        agent=GeminiCLIAgent,
        models={
            "agent": [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                # Add or swap models you have access to.
            ],
        },
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",
    )

    results = selector.select_best(parallel=False)
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest model: {best}")
