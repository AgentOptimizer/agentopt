"""
Example: OpenClaw with agentopt.

Find the best LLM model for your OpenClaw tasks. agentopt automatically
patches OpenClaw's config to route LLM calls through its proxy, tracks
token usage, latency, and cost, then selects the best model.

Prerequisites:
    1. pip install agentopt-py
    2. npm install -g openclaw   (OpenClaw CLI)
    3. Set up at least one provider in OpenClaw (e.g. ``openclaw onboard``)
    4. Set ANTHROPIC_API_KEY (or other provider keys) in your environment

Usage:
    python examples/openclaw_example.py
"""

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

from agentopt import ModelSelector
from openclaw_agent import OpenClawAgent

# ---------------------------------------------------------------------------
# Step 1: Evaluation dataset — (input_data, expected_output) pairs.
#
# Replace these with your actual evaluation tasks. Each entry is
# (prompt, expected_answer) — the eval function compares the model's
# output against the expected answer.
# ---------------------------------------------------------------------------

dataset = [
    ("What is 2 + 2?", "4"),
    ("What is the capital of France?", "Paris"),
    ("What color is the sky on a clear day?", "blue"),
    # Add more evaluation pairs here. We recommend 10-20 for development,
    # 50+ for production model selection.
]


# ---------------------------------------------------------------------------
# Step 2: Evaluation function — score the model's response.
#
# Returns 1.0 if the expected answer appears in the response, 0.0 otherwise.
# Customize this for your use case (e.g. exact match, LLM-as-judge, etc.)
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    actual_str = str(actual)
    if "FAILED" in actual_str:
        print(f"  [debug] sample FAILED: {actual_str[:300]}")
        return 0.0
    matched = expected.lower() in actual_str.lower()
    if not matched:
        print(f"  [debug] no match — expected={expected!r}, got={actual_str[:300]!r}")
    return 1.0 if matched else 0.0


# ---------------------------------------------------------------------------
# Step 3: Run model selection.
#
# agentopt tries each model across all tasks and ranks by accuracy,
# latency, and cost. LLM calls are tracked automatically via the proxy.
#
# IMPORTANT: Use parallel=False — OpenClaw config patching is not
# safe for concurrent evaluation (single shared config file).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify openclaw is available
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

    # Verify OpenClaw has been onboarded (config file exists with at least one provider)
    config_path = os.path.expanduser("~/.openclaw/openclaw.json")
    if not os.path.exists(config_path):
        print(
            f"Error: OpenClaw config not found at {config_path}.\n"
            "Run `openclaw onboard` to set up at least one provider first."
        )
        sys.exit(1)

    selector = ModelSelector(
        agent=OpenClawAgent,
        models={
            "agent": [
                "anthropic/claude-haiku-4-5",
                "anthropic/claude-sonnet-4-6",
                # "openai/gpt-4o-mini",  # Requires OpenAI configured in OpenClaw + stream_options for token tracking
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
