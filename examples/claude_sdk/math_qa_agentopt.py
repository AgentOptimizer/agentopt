"""Claude SDK math QA with AgentOpt model selection.

Architecture at a glance:
- ClaudeChat wraps the Anthropic client and exposes `.model` + `.invoke`.
- ModelProxy sits around ClaudeChat so ModelSelector can hot-swap model names.
- ModelSelector drives evaluation over a dataset via `invoke_fn`.
"""

from agentopt import ModelProxy, ModelSelector
from anthropic import Anthropic
from examples.sdk_shared import ClaudeChat, eval_fn, load_jsonl_dataset


def main(dataset_dir: str = "examples/datasets") -> None:
    dataset = load_jsonl_dataset(dataset_dir)
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
    print(f"Best Claude model: {results.get_best()}")


if __name__ == "__main__":
    main()
