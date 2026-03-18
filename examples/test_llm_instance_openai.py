"""
LLM-instance model-swap smoke test (OpenAI).

Verifies:
1) OPENAI_API_KEY is valid (direct API probe)
2) agentopt evaluates prebuilt ChatOpenAI instances as candidates
3) model selection completes with real API calls
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI
from openai import OpenAI

from agentopt import BruteForceModelSelector


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _eval_fn(expected: str, actual: Any) -> float:
    text = _to_text(actual)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return 1.0 if digits == expected else 0.0
    return 1.0 if expected.lower() in text.lower() else 0.0


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Load .env first: "
            "`set -a; source ../.env; set +a`."
        )

    probe = OpenAI(api_key=api_key).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply exactly with API_OK"}],
        max_tokens=5,
        temperature=0,
    )
    print(
        f"[openai probe] completion_id={probe.id} "
        f"output={(probe.choices[0].message.content or '').strip()!r}"
    )

    dataset: List[Tuple[Dict[str, str], str]] = [
        ({"question": "What is 3 + 5? Return only the number."}, "8"),
        ({"question": "What is 10 - 3? Return only the number."}, "7"),
    ]

    models = {
        "agent": [
            ChatOpenAI(model="gpt-4o-mini", temperature=0),
            ChatOpenAI(model="gpt-4.1", temperature=0),
        ]
    }

    def agent_maker(model_map: Dict[str, Any]):
        llm = model_map["agent"]
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        print(f"[instance selected] {model_name}")

        def run(input_data: Dict[str, str]) -> str:
            response = llm.invoke(input_data["question"])
            text = _to_text(response)
            print(f"[instance output] {model_name}: {text!r}")
            return text

        return run

    selector = BruteForceModelSelector(
        agent_fn=agent_maker,
        models=models,
        eval_fn=_eval_fn,
        dataset=dataset,
    )
    results = selector.select_best()
    results.print_summary()
    print(f"best={results.get_best_combo()}")


if __name__ == "__main__":
    main()
