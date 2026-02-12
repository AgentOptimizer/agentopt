"""OpenAI Agents SDK math QA with AgentOpt model selection."""

from typing import Iterable, Sequence
from types import SimpleNamespace

from agents import Agent, Runner

from agentopt import ModelProxy, ModelSelector
from .utils import AgentFactoryRunner, eval_fn, load_jsonl_dataset


def build_agent(model: str = "gpt-4o-mini") -> Agent:
    return Agent(
        name="Math QA Agent",
        model=model,
        instructions="Answer the user's math question concisely.",
    )


def run_agent(agent: Agent, question: str) -> str:
    result = Runner.run_sync(agent, question)
    return result.final_output if hasattr(result, "final_output") else str(result)


def main(
    dataset_dir: str = "examples/datasets",
    candidate_models: Sequence[str] | Iterable[str] = ("gpt-4o-mini", "gpt-4o"),
) -> None:
    dataset = load_jsonl_dataset(dataset_dir)

    proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    runner = AgentFactoryRunner(proxy, build_agent, run_agent)

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        agent=runner,
    )

    results = selector.select_best()
    print(f"Best OpenAI model: {results.get_best()}")


if __name__ == "__main__":
    main()
