"""General pattern: wrap Claude Agent SDK async query workflow with AgentOpt."""

from typing import Callable, Iterable, Sequence

from claude_agent_sdk import ClaudeAgentOptions
from agentopt import ModelProxy, ModelSelector
from .utils import AgentFactoryRunner, eval_fn, load_jsonl_dataset, run_query_sync


def select_best_claude_agent(
    agent_factory: Callable[[str], object],
    candidate_models: Sequence[str] | Iterable[str],
    dataset_path: str = "examples/datasets/math_problems.jsonl",
):
    """Evaluate the same Claude Agent SDK setup across models using AgentOpt."""

    dataset = load_jsonl_dataset(dataset_path)

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
    best = results.get_best()
    print(f"Best Claude model: {best}")

    if best:
        proxy.set_model(best.model_name)
        final_agent = ClaudeAgentOptions(model=proxy.get_model())
        sample = dataset[0][0]["input"]
        final_output = run_query_sync(sample, final_agent.model)
        print(f"Sample run with best model ({best.model_name}): {final_output}")

    return results


if __name__ == "__main__":
    select_best_claude_agent(
        agent_factory=lambda model: SimpleNamespace(model=model),
        candidate_models=["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
        dataset_path="examples/datasets/math_problems.jsonl",
    )
