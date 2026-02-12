"""Claude Agent SDK math QA with AgentOpt model selection (async query API)."""

from typing import Iterable, Sequence

from claude_agent_sdk import ClaudeAgentOptions
from agentopt import ModelProxy, ModelSelector
from .utils import AgentFactoryRunner, eval_fn, load_jsonl_dataset, run_query_sync


def main(
    dataset_dir: str = "examples/datasets",
    candidate_models: Sequence[str] | Iterable[str] = ("claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"),
) -> None:
    dataset = load_jsonl_dataset(dataset_dir)

    proxy = ModelProxy(ClaudeAgentOptions(model="claude-3-5-haiku-latest"))
    runner = AgentFactoryRunner(
        proxy,
        agent_factory=lambda model: ClaudeAgentOptions(model=model),
        run_fn=lambda agent_opts, question: run_query_sync(question, agent_opts.model),
    )

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        agent=runner,
    )

    results = selector.select_best()
    print(f"Best Claude model: {results.get_best()}")


if __name__ == "__main__":
    main()
