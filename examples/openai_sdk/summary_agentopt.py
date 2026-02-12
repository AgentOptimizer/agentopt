"""OpenAI Agents SDK summarization with AgentOpt model selection."""

from types import SimpleNamespace
from typing import Iterable, Sequence

from openai import OpenAI

from agentopt import ModelProxy, ModelSelector
from examples.sdk_shared import AgentFactoryRunner, eval_fn, small_summary_dataset


def build_agent(model: str = "gpt-4o-mini"):
    client = OpenAI()
    return client.agents.create(
        name="Summarizer",
        model=model,
        instructions="Summarize the user message in one sentence.",
    )


def run_agent(agent, text: str) -> str:
    client = OpenAI()
    result = client.agents.runs.create_and_poll(agent_id=agent.id, input=text)
    if hasattr(result, "output") and result.output:
        return result.output[0].content[0].text
    return str(result)


def main(candidate_models: Sequence[str] | Iterable[str] = ("gpt-4o-mini", "gpt-4o")) -> None:
    dataset = small_summary_dataset()

    proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    runner = AgentFactoryRunner(proxy, build_agent, run_agent)

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        agent=runner,
    )

    results = selector.select_best()
    print(f"Best OpenAI model (summaries): {results.get_best()}")


if __name__ == "__main__":
    main()
