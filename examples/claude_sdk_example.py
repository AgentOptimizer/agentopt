"""Claude Agent SDK examples with AgentOpt model selection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

from claude_agent_sdk import ClaudeAgentOptions, query

from agentopt import ModelProxy, ModelSelector

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Math QA with AgentOpt
# ---------------------------------------------------------------------------


def math_qa_agentopt(
    candidate_models: Sequence[str] | Iterable[str] = (
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
    ),
    dataset_dir: str = "examples",
) -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    proxy = ModelProxy(ClaudeAgentOptions(model="claude-3-5-haiku-latest"))

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=lambda input_data: run_query_sync(input_data["input"], proxy.model),
    )

    results = selector.select_best()
    print(f"Best Claude model: {results.get_best()}")


# ---------------------------------------------------------------------------
# Math QA baseline (no AgentOpt)
# ---------------------------------------------------------------------------


def math_qa_baseline(dataset_dir: str = "examples") -> None:
    dataset = load_jsonl_dataset(dataset_dir)

    for input_data, expected in dataset:
        answer = run_query_sync(input_data["input"], "claude-3-5-haiku-latest")
        print(f"Q: {input_data['input']}\nA: {answer}\nExpected: {expected}\n")


if __name__ == "__main__":
    math_qa_agentopt()
