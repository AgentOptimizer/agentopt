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
from agentopt.integrations.openclaw import OpenClawAgent

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
    if "FAILED" in str(actual):
        return 0.0
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


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
            capture_output=True, timeout=10,
            env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Error: openclaw CLI not found. Install with: npm install -g openclaw")
        sys.exit(1)

    selector = ModelSelector(
        agent=OpenClawAgent,
        models={
            "agent": [
                "anthropic/claude-sonnet-4-6",
                # Add more models to compare:
                # "anthropic/claude-haiku-4-5",
                # "openai/gpt-4o-mini",  # Note: OpenAI token tracking requires stream_options
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
