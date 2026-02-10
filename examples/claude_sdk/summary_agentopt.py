"""Claude SDK summarization with AgentOpt model selection.

Architecture: ClaudeChat wraps the Anthropic client and surfaces `.model` + `.invoke`.
ModelProxy keeps that wrapper stable while ModelSelector swaps model names during
evaluation via `invoke_fn`.
"""

from agentopt import ModelProxy, ModelSelector
from anthropic import Anthropic
from examples.sdk_shared import ClaudeChat, eval_fn, small_summary_dataset


def main() -> None:
    dataset = small_summary_dataset()
    proxy = ModelProxy(ClaudeChat(Anthropic(), model="claude-3-5-haiku-latest"))

    selector = ModelSelector(
        models={
            proxy: [
                "claude-3-5-haiku-latest",
                "claude-3-5-sonnet-latest",
            ]
        },
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=lambda payload: proxy.invoke(payload),
    )

    results = selector.select_best()
    print(f"Best Claude model (summaries): {results.get_best()}")


if __name__ == "__main__":
    main()
