"""OpenAI SDK summarization with AgentOpt model selection.

Architecture: OpenAIChat wraps the SDK client and exposes `.model` + `.invoke`.
ModelProxy keeps the same wrapper instance stable while ModelSelector iterates
through candidate model names via `invoke_fn`.
"""

from agentopt import ModelProxy, ModelSelector
from examples.sdk_shared import eval_fn, small_summary_dataset, OpenAIChat
from openai import OpenAI


def main() -> None:
    dataset = small_summary_dataset()
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
    print(f"Best OpenAI model (summaries): {results.get_best()}")


if __name__ == "__main__":
    main()
