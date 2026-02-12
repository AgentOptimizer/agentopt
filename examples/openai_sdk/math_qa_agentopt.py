"""OpenAI Agents SDK math QA with AgentOpt model selection.

Architecture: OpenAIChat wraps the SDK client and exposes `.model` + `.invoke`.
ModelProxy sits around that wrapper so ModelSelector can swap model names while
reusing the same agent wiring via `invoke_fn`.
"""

from agentopt import ModelProxy, ModelSelector
from examples.sdk_shared import eval_fn, load_jsonl_dataset, OpenAIChat
from openai import OpenAI


def main(dataset_dir: str = "examples/datasets") -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    proxy = ModelProxy(OpenAIChat(OpenAI(), model="gpt-4o-mini"))

    selector = ModelSelector(
        models={
            proxy: [
                "gpt-4o-mini",
                "gpt-4o",
            ]
        },
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=lambda payload: proxy.invoke(payload),
    )

    results = selector.select_best()
    print(f"Best OpenAI model: {results.get_best()}")


if __name__ == "__main__":
    main()
