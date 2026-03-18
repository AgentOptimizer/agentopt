"""
Model-swap test for OpenAI + Claude using real API calls.

What this validates:
1) Provider API key works (direct probe call).
2) agentopt evaluates multiple LLM instance candidates.
3) Candidate swapping happens (prints selected instance per evaluation).

Usage:
    cd agentopt-new
    source ../.venv/bin/activate
    set -a; source ../.env; set +a
    PYTHONPATH=agentopt/src:agentproxy/src python examples/test_model_swap_openai_claude.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import httpx
from langchain_openai import ChatOpenAI
from openai import OpenAI

from agentopt import BruteForceModelSelector


@dataclass
class ProviderProof:
    provider: str
    probe_id: str
    call_ids: List[str]
    best_combo: Dict[str, str] | None


class ClaudeHTTPInstance:
    """Minimal Claude model instance with .invoke(prompt) for testing."""

    def __init__(self, model_name: str, api_key: str, timeout: float = 20.0) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._timeout = timeout
        self.last_request_id: str = ""
        self.last_status: int = 0

    def invoke(self, prompt: str) -> str:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model_name,
                "max_tokens": 48,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self._timeout,
        )
        request_id = response.headers.get("request-id")
        self.last_request_id = request_id or ""
        self.last_status = response.status_code
        if response.status_code >= 400:
            raise RuntimeError(
                f"Claude request failed ({self.model_name}) "
                f"status={response.status_code}, request_id={request_id}, "
                f"body={response.text}"
            )
        payload = response.json()
        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ).strip()
        print(
            f"[claude http] model={self.model_name} "
            f"status={response.status_code} request_id={request_id}"
        )
        return text


def _question_prompt(question: str) -> str:
    return f"Answer with only the final short answer.\nQuestion: {question}"


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content).strip()


def _eval_fn(expected: str, actual: Any) -> float:
    text = _response_to_text(actual)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return 1.0 if digits == expected else 0.0
    return 1.0 if expected.lower() in text.lower() else 0.0


def _run_selector(provider: str, candidates: List[Any]) -> Tuple[Dict[str, str] | None, List[str]]:
    dataset: List[Tuple[Dict[str, str], str]] = [
        ({"question": "What is 3 + 5?"}, "8"),
    ]
    call_ids: List[str] = []

    def agent_maker(models: Dict[str, Any]):
        llm = models["agent"]
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        print(f"[{provider} selected] type={type(llm).__name__} model={model_name}")

        def run(input_data: Dict[str, str]) -> str:
            response = llm.invoke(_question_prompt(input_data["question"]))
            output = _response_to_text(response)
            if provider == "openai":
                # For LangChain OpenAI responses, usage metadata is direct proof
                # that a real completion happened for this candidate call.
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    in_t = usage.get("input_tokens", 0)
                    out_t = usage.get("output_tokens", 0)
                    call_id = f"{model_name}:{in_t}/{out_t}"
                    call_ids.append(call_id)
                    print(f"[openai call] model={model_name} tokens={in_t}/{out_t}")
            elif provider == "claude" and isinstance(llm, ClaudeHTTPInstance):
                if llm.last_request_id:
                    call_ids.append(llm.last_request_id)
            return output

        return run

    selector = BruteForceModelSelector(
        agent_fn=agent_maker,
        models={"agent": candidates},
        eval_fn=_eval_fn,
        dataset=dataset,
    )
    results = selector.select_best()
    print(f"[{provider}] best={results.get_best_combo()}")
    return results.get_best_combo(), call_ids


def _run_openai() -> ProviderProof | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[openai] skipped (OPENAI_API_KEY not set)")
        return None

    print(f"[openai] key detected: {api_key[:7]}...")
    probe = OpenAI(api_key=api_key).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply exactly with API_OK"}],
        max_tokens=5,
        temperature=0,
    )
    output = (probe.choices[0].message.content or "").strip()
    print(f"[openai probe] completion_id={probe.id} output={output!r}")

    candidates = [
        ChatOpenAI(model="gpt-4o-mini", temperature=0),
        ChatOpenAI(model="gpt-4.1", temperature=0),
    ]
    best_combo, call_ids = _run_selector("openai", candidates)
    if not call_ids:
        raise RuntimeError("OpenAI selector run completed but no call token evidence found.")
    return ProviderProof(
        provider="openai",
        probe_id=probe.id,
        call_ids=call_ids,
        best_combo=best_combo,
    )


def _run_claude() -> ProviderProof | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[claude] skipped (ANTHROPIC_API_KEY not set)")
        return None

    print(f"[claude] key detected: {api_key[:7]}...")
    probe = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply exactly with API_OK"}],
        },
        timeout=20.0,
    )
    if probe.status_code >= 400:
        raise RuntimeError(
            f"Claude probe failed: status={probe.status_code}, body={probe.text}"
        )
    probe_payload = probe.json()
    probe_text = "".join(
        block.get("text", "")
        for block in probe_payload.get("content", [])
        if block.get("type") == "text"
    ).strip()
    print(
        f"[claude probe] message_id={probe_payload.get('id')} "
        f"request_id={probe.headers.get('request-id')} output={probe_text!r}"
    )
    probe_request_id = probe.headers.get("request-id") or ""
    if not probe_request_id:
        raise RuntimeError("Claude probe response missing request_id header.")

    candidates = [
        ClaudeHTTPInstance("claude-haiku-4-5-20251001", api_key),
        ClaudeHTTPInstance("claude-sonnet-4-5-20250929", api_key),
    ]
    best_combo, call_ids = _run_selector("claude", candidates)
    if not call_ids:
        raise RuntimeError("Claude selector run completed but no request_id evidence found.")
    return ProviderProof(
        provider="claude",
        probe_id=probe_request_id,
        call_ids=call_ids,
        best_combo=best_combo,
    )


def main() -> None:
    openai_proof = _run_openai()
    claude_proof = _run_claude()
    if openai_proof is None and claude_proof is None:
        raise RuntimeError(
            "No provider keys found. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY."
        )

    print("\n=== API CALL PROOF ===")
    if openai_proof is not None:
        print(f"[openai] probe_completion_id={openai_proof.probe_id}")
        print(f"[openai] selector_call_evidence={openai_proof.call_ids}")
        print(f"[openai] best={openai_proof.best_combo}")
    if claude_proof is not None:
        print(f"[claude] probe_request_id={claude_proof.probe_id}")
        print(f"[claude] selector_request_ids={claude_proof.call_ids}")
        print(f"[claude] best={claude_proof.best_combo}")
    print("PASS: provider model-swap tests completed with real API calls.")


if __name__ == "__main__":
    main()
