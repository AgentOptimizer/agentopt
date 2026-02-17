"""
Manual sanity test: confirm ModelSelector calls proxy.set_model() for Claude SDK.
Claude-only to avoid OpenAI Agents SDK imports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Use in-repo agentopt (with ModelProxy.set_model)
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from agentopt import ModelProxy, BruteForceModelSelector  # noqa: E402


def run_claude():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[claude] skipped (ANTHROPIC_API_KEY not set)")
        return
    print(f"[claude] key detected (starts with {api_key[:6]}…)")
    # Quick connectivity probe (no caching) to fail fast on bad keys.
    import httpx
    candidates = [
        "claude-opus-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ]
    print(f"[claude] using hardcoded candidates: {candidates}")

    original_set_model = ModelProxy.set_model

    def noisy_set_model(self, model):
        print(f"[claude set_model] -> {model}")
        return original_set_model(self, model)

    ModelProxy.set_model = noisy_set_model  # type: ignore

    # Store model name in a SimpleNamespace so set_model(str) can update it.
    from types import SimpleNamespace

    proxy = ModelProxy(SimpleNamespace(model=candidates[0]))
    dataset = [({"input": "What is 3 + 5?"}, "8")]
    system_prompt = "Answer with just the final number."

    def invoke_fn(_input):
        model_name = proxy.get_model().model
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model_name,
                    "max_tokens": 32,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": _input["input"]}],
                        },
                    ],
                },
                timeout=15.0,
            )
            if resp.status_code >= 400:
                print(f"[invoke error body] {model_name}: {resp.text}")
                output = ""
            else:
                print(
                    f"[claude http] {model_name} status={resp.status_code} request_id={resp.headers.get('request-id')}"
                )
                output = resp.json()["content"][0]["text"]
        except Exception as exc:
            print(f"[invoke error] {model_name}: {exc}")
            output = ""
        else:
            print(f"[claude output] {model_name}: {output!r}")
        return {"model": model_name, "output": output}

    def eval_fn(expected: str, actual) -> bool:
        out = (actual.get("output") or "").strip()
        print(f"[claude eval] expected={expected!r}, got={out!r}")
        # Handle numeric answers like "8" possibly with punctuation/text
        try:
            digits = "".join(ch for ch in out if ch.isdigit())
            if digits:
                return int(expected) == int(digits)
        except Exception:
            pass
        return expected in out

    selector = BruteForceModelSelector(
        models={proxy: candidates},
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
    )

    results = selector.select_best()
    print("[claude] results:")
    for r in results:
        print(f"  {r.model_name}: best={r.is_best}")


if __name__ == "__main__":
    run_claude()
