"""Example: Track Gemini CLI calls through AgentOpt proxy.

Prerequisites:
    1. Install Gemini CLI and authenticate it
    2. pip install agentopt-py

This example demonstrates subprocess tracking. It routes Gemini CLI HTTPS traffic
through a session-scoped proxy and records model/tokens/latency in CallRecords.
"""

import os
import subprocess

from agentopt.proxy import LLMTracker


def main() -> None:
    tracker = LLMTracker(cache=False)
    tracker.start()

    try:
        with tracker.track(data_id="dp_1", combo_id="gemini-cli") as session:
            env = {**os.environ, **tracker.get_session_env(session)}
            subprocess.run(
                ["gemini", "prompt", "Write one sentence about model selection."],
                env=env,
                check=True,
            )

        records = tracker.get_records(data_id="dp_1", combo_id="gemini-cli")
        print(f"captured calls: {len(records)}")
        for i, r in enumerate(records, start=1):
            print(
                f"[{i}] model={r.model} "
                f"prompt_tokens={r.prompt_tokens} "
                f"completion_tokens={r.completion_tokens} "
                f"latency={r.latency_seconds:.2f}s"
            )
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
