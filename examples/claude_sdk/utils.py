from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import asyncio

from agentopt import ModelProxy  # type: ignore
from claude_agent_sdk import query, ClaudeAgentOptions


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


def eval_fn(expected: str, actual: Any) -> bool:
    actual_text = actual.get("output") if isinstance(actual, dict) else str(actual)
    return expected.lower() in str(actual_text).lower()


async def _run_query_async(prompt: str, model: str, **opts) -> str:
    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(model=model, **opts),
    ):
        if hasattr(message, "result"):
            result_text = message.result
    return result_text


def run_query_sync(prompt: str, model: str, **opts) -> str:
    return asyncio.run(_run_query_async(prompt, model, **opts))


class AgentFactoryRunner:
    """Adapter so ModelSelector can call `.invoke` without a custom invoke_fn."""

    def __init__(self, proxy: ModelProxy, agent_factory, run_fn) -> None:
        self.proxy = proxy
        self.agent_factory = agent_factory
        self.run_fn = run_fn

    def invoke(self, payload: dict[str, str]) -> Any:
        question = payload.get("input", payload)
        model_obj = self.proxy.get_model()
        model_value = getattr(model_obj, "model", model_obj)
        agent_cfg = self.agent_factory(model_value)
        return self.run_fn(agent_cfg, question)
