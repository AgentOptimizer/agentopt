"""Example: use an LLM router to suggest model candidates before selection."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from agentopt import BruteForceModelSelector, LLMModelSuggester, ModelProxy


def eval_fn(expected: str, actual: str) -> bool:
    return expected.lower() in str(actual).lower()


def main() -> None:
    all_candidates = [
        ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        ChatOpenAI(model="gpt-4o", temperature=0.0),
        ChatOpenAI(model="gpt-4.1", temperature=0.0),
    ]

    router = LLMModelSuggester(router_model="gpt-4o-mini")
    shortlisted = router.shortlist_candidates(
        task="Short arithmetic QA with strict answer format and low latency preference.",
        candidates=all_candidates,
        top_k=2,
        context={"prefer_low_latency": True, "budget_sensitive": True},
    )
    print("Shortlisted:", [getattr(m, "model_name", None) or getattr(m, "model", "?") for m in shortlisted])

    proxy = ModelProxy(shortlisted[0])

    def invoke_fn(input_data: dict[str, str]) -> str:
        response = proxy.invoke(input_data["input"])
        return response.content if hasattr(response, "content") else str(response)

    dataset = [
        ({"input": "Answer with only the integer: 2+2"}, "4"),
        ({"input": "Answer with only the integer: 10-3"}, "7"),
    ]

    selector = BruteForceModelSelector(
        models={proxy: shortlisted},
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
    )
    results = selector.select_best(parallel=False)
    print("Best:", results.get_best())


if __name__ == "__main__":
    main()
