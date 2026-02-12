"""Shared utilities for SDK-only examples.

These helpers keep OpenAI/Claude examples consistent:
- load_jsonl_dataset: JSONL -> list[(input_dict, expected_answer)]
- small_summary_dataset: tiny inline dataset for summarization tasks
- eval_fn: tolerant check for expected answer containment
- OpenAIChat / ClaudeChat: thin wrappers exposing a `.model` attribute and `.invoke`
  so they can be wrapped by ModelProxy for AgentOpt model selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

from anthropic import Anthropic
from openai import OpenAI

class AgentFactoryRunner:
    """
    Minimal adapter so ModelSelector can call `.invoke()` without custom invoke_fn.

    It rebuilds the agent/config for each evaluation using the current model name
    stored on the provided ModelProxy.
    """

    def __init__(self, proxy, agent_factory, run_fn) -> None:
        self.proxy = proxy
        self.agent_factory = agent_factory
        self.run_fn = run_fn

    def invoke(self, payload: dict[str, str]) -> Any:
        question = payload.get("input", payload)
        agent = self.agent_factory(self.proxy.get_model())
        return self.run_fn(agent, question)


def load_jsonl_dataset(dataset_dir: str) -> list[Tuple[dict[str, str], str]]:
    dataset_path = Path(dataset_dir)
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in: {dataset_dir}")

    tasks: list[Tuple[dict[str, str], str]] = []
    with open(jsonl_files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks


def small_summary_dataset() -> list[Tuple[dict[str, str], str]]:
    return [
        (
            {"input": "Summarize: AgentOpt helps swap LLMs without rebuilding agents."},
            "swap llms without rebuild",
        ),
        (
            {"input": "Summarize: Claude 3.5 Sonnet excels at long-context reasoning."},
            "long-context reasoning",
        ),
    ]


def eval_fn(expected: str, actual: Any) -> bool:
    if isinstance(actual, dict):
        actual_text = actual.get("output") or str(actual)
    else:
        actual_text = str(actual)
    return expected.lower() in actual_text.lower()


class OpenAIChat:
    """Thin wrapper around OpenAI client with mutable model field."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model

    def invoke(self, input_data: dict[str, str]) -> str:
        prompt = input_data.get("input", "")
        # Prefer the Responses/Runner-style API (more general), fall back to chat.completions.
        if hasattr(self.client, "responses"):
            resp = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=256,
            )
            # New SDK exposes `output_text`; otherwise flatten first block.
            if hasattr(resp, "output_text") and resp.output_text is not None:
                return resp.output_text
            if getattr(resp, "output", None):
                block = resp.output[0]
                if getattr(block, "content", None):
                    content = block.content[0]
                    text = getattr(content, "text", None)
                    if text:
                        return text
            return str(resp)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
        )
        return response.choices[0].message.content or ""


class ClaudeChat:
    """Anthropic Messages API wrapper compatible with ModelProxy."""

    def __init__(self, client: Anthropic, model: str = "claude-3-5-haiku-latest") -> None:
        self.client = client
        self.model = model

    def invoke(self, input_data: dict[str, str]) -> str:
        prompt = input_data.get("input", "")
        message = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        if message.content and hasattr(message.content[0], "text"):
            return message.content[0].text
        return str(message)
