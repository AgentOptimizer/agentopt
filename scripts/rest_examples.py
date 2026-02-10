"""Quick runner for SDK examples (acts like a REST test harness placeholder).

Usage:
    uv run python scripts/rest_examples.py

It runs one baseline and one AgentOpt example for both OpenAI and Claude SDKs.
Adjust or extend as needed.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "examples/openai_sdk/math_qa_baseline.py"],
    [sys.executable, "examples/openai_sdk/math_qa_agentopt.py"],
    [sys.executable, "examples/claude_sdk/math_qa_baseline.py"],
    [sys.executable, "examples/claude_sdk/math_qa_agentopt.py"],
]


def run_cmd(cmd):
    print(f"\n=== Running: {' '.join(cmd)} ===")
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        print(f"Command failed with code {completed.returncode}")


def main():
    for cmd in COMMANDS:
        run_cmd(cmd)


if __name__ == "__main__":
    main()
