"""General pattern: wrap an OpenAI Agents SDK workflow with AgentOpt.

- `build_agent(model)` constructs your Agents SDK agent (tools/prompt unchanged).
- `run_agent(agent, question)` runs it once via Runner.
- ModelProxy holds the model name; AgentFactoryRunner exposes `.invoke`, so you don't
  need a custom invoke_fn—ModelSelector calls it directly.
"""

from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

from openai import OpenAI

from agentopt import ModelProxy, ModelSelector
from .utils import AgentFactoryRunner, eval_fn, load_jsonl_dataset


# --- Your existing Agents SDK factory ---

def build_agent(model: str = "gpt-4o-mini"):
    client = OpenAI()
    return client.agents.create(
        name="Budget Helper",
        model=model,
        instructions=(
            "You are a budgeting assistant. "
            "Call list_expenses to see USD amounts, then convert_to_eur for totals. "
            "Return both USD and EUR totals and highlight the top two spend categories."
        ),
    )


def run_agent(agent, question: str) -> str:
    """Execute a single user message against an Agents SDK Agent via Runner."""
    client = OpenAI()
    result = client.agents.runs.create_and_poll(agent_id=agent.id, input=question)
    if hasattr(result, "output") and result.output:
        return result.output[0].content[0].text
    return str(result)


# --- AgentOpt wiring ---

def select_best_openai_agent(
    agent_factory: Callable[[str], object],
    candidate_models: Sequence[str] | Iterable[str],
    dataset_path: str = "examples/datasets/math_problems.jsonl",
):
    """Evaluate the same Agents SDK setup across models using AgentOpt."""

    dataset = load_jsonl_dataset(dataset_path)

    # Proxy tracks the active model string; ModelSelector will mutate it.
    proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    runner = AgentFactoryRunner(proxy, agent_factory, run_agent)

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        agent=runner,
    )

    results = selector.select_best()
    best = results.get_best()
    print(f"Best model: {best}")

    # Optionally rebuild agent with the winning model for production use
    if best:
        proxy.set_model(best.model_name)
        final_agent = agent_factory(proxy.get_model())
        sample = dataset[0][0]["input"]
        final_output = run_agent(final_agent, sample)
        print(f"Sample run with best model ({best.model_name}): {final_output}")

    return results


if __name__ == "__main__":
    select_best_openai_agent(
        agent_factory=build_agent,
        candidate_models=["gpt-4o-mini", "gpt-4o"],
        dataset_path="examples/datasets/math_problems.jsonl",
    )
