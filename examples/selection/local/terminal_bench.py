"""
Example: Terminal Bench (tbench) with agentopt.

Find the best LLM model for Terminal Bench tasks. The agent wraps the
``tb run`` CLI as a subprocess — agentopt systematically evaluates each
candidate model across a set of tbench tasks and ranks them by pass rate,
latency, and cost.

Token usage, latency, and cost are tracked automatically: while a
``track()`` scope is active, agentopt's subprocess patch transparently
injects ``HTTPS_PROXY`` + the merged CA bundle path into any child
process spawned via ``subprocess.run`` / ``Popen`` — the user's
``run()`` method stays free of any agentopt imports.

Prerequisites:
    1. pip install agentopt-py
    2. uv tool install terminal-bench   (requires Docker and git)
    3. Docker must be running
    4. Set API keys per LiteLLM docs:
       - ANTHROPIC_API_KEY for anthropic/ models
       - OPENAI_API_KEY for openai/ models
"""

import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

from agentopt import ModelSelector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TB_DATASET = "terminal-bench-core==0.1.1"
TB_AGENT = "terminus"
TB_TIMEOUT = 600  # seconds per task


# ---------------------------------------------------------------------------
# Step 1: Define your agent class.
# __init__(models) receives a dict like {"agent": "anthropic/claude-sonnet-4-20250514"}.
# run(input_data) runs the agent on a single datapoint and returns the output.
# ---------------------------------------------------------------------------


class TerminalBenchAgent:
    """Wraps ``tb run`` as a subprocess for agentopt model selection."""

    def __init__(self, models):
        self.model = models["agent"]

    def run(self, input_data):
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
            input_data,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=TB_TIMEOUT,
            )
            return result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            return "FAILED: timeout"
        except FileNotFoundError:
            return "FAILED: tb command not found — run: uv tool install terminal-bench"


# ---------------------------------------------------------------------------
# Step 2: Evaluation dataset — (task_id, expected_result) pairs.
# Discover available tasks with: tb tasks list --dataset terminal-bench-core==head
# ---------------------------------------------------------------------------

dataset = [
    ("hello-world", "pass"),
    # Add more task_ids here. We recommend 10-20 tasks for development,
    # 50+ for production model selection.
]


# ---------------------------------------------------------------------------
# Step 3: Evaluation function — parse tb output for pass/fail.
# Run `tb run ... --task-id hello-world` manually first to verify the
# output format matches this parsing logic.
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    output = str(actual).lower()
    if "pass" in output or "passed" in output:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Step 4: Run model selection.
# AgentOpt tries each model across all tasks and ranks by pass rate,
# latency, and cost.  Subprocess LLM calls are tracked automatically —
# no env-var plumbing needed inside the agent.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify Docker daemon is running — `tb run` launches a container per task
    # and fails fast (~5s) without a clear error if Docker isn't reachable,
    # silently scoring every task 0%.
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(
                "Error: Docker daemon is not running.\n"
                "Terminal Bench launches a Docker container per task. "
                "Start Docker Desktop (or `colima start`) and try again.\n"
                f"\ndocker info said:\n{result.stderr.strip()[:300]}"
            )
            sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: docker CLI not found. "
            "Install Docker Desktop (https://www.docker.com/products/docker-desktop) "
            "or another Docker-compatible runtime, then try again."
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(
            "Error: `docker info` timed out — the Docker daemon may be hung. Restart Docker and try again."
        )
        sys.exit(1)

    selector = ModelSelector(
        agent=TerminalBenchAgent,
        models={"agent": ["openai/gpt-4o", "openai/gpt-4o-mini",],},
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",
    )

    # Sequential execution: each tb run launches a Docker container.
    results = selector.select_best(parallel=False)
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest model: {best}")
