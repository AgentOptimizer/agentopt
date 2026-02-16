"""OpenAI Agents SDK examples with AgentOpt model selection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence, Tuple

from agents import Agent, Runner

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


# ---------------------------------------------------------------------------
# Math QA with AgentOpt
# ---------------------------------------------------------------------------


def math_qa_agentopt(
    candidate_models: Sequence[str] | Iterable[str] = ("gpt-4o-mini", "gpt-4o"),
    dataset_dir: str = "examples",
) -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    # Build the Agent once with a ModelProxy so ModelSelector can hot-swap
    # the model via proxy.set_model(...) without rebuilding the agent.
    proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    agent = Agent(
        name="Math QA",
        model=proxy,
        instructions="Answer the user's math question concisely.",
    )

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        # Runner.run_sync reuses the constructed agent; ModelSelector mutates
        # proxy between evaluations so each run uses the current candidate model.
        invoke_fn=lambda input_data: Runner.run_sync(agent, input_data["input"]).final_output,
    )

    results = selector.select_best()
    print(f"Best OpenAI model: {results.get_best()}")


# ---------------------------------------------------------------------------
# Math QA baseline (no AgentOpt)
# ---------------------------------------------------------------------------


def math_qa_baseline(dataset_dir: str = "examples") -> None:
    dataset = load_jsonl_dataset(dataset_dir)

    for input_data, expected in dataset:
        result = Runner.run_sync(
            Agent(
                name="Math QA",
                model="gpt-4o-mini",
                instructions="Answer the user's math question concisely.",
            ),
            input_data["input"],
        )
        answer = result.final_output if hasattr(result, "final_output") else str(result)
        print(f"Q: {input_data['input']}\nA: {answer}\nExpected: {expected}\n")


if __name__ == "__main__":
    math_qa_agentopt()
