"""General pattern: wrap a plain OpenAI SDK chat workflow with AgentOpt.

- `build_client(model)` returns a chat client + model pair.
- `run_chat(agent, question)` issues a chat.completions call.
- ModelProxy holds the model name; AgentFactoryRunner exposes `.invoke`, so you don't
  need to write a custom invoke_fn—ModelSelector calls it directly.
"""

from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

from openai import OpenAI

from agentopt import ModelProxy, ModelSelector
from examples.sdk_shared import AgentFactoryRunner, eval_fn, load_jsonl_dataset


# --- Your existing client/chat factory (vanilla OpenAI SDK) ---

def build_client(model: str = "gpt-4o-mini"):
    """Return a SimpleNamespace with client + model; swap model via ModelProxy."""
    return SimpleNamespace(client=OpenAI(), model=model)


def run_chat(agent, question: str) -> str:
    """Execute a single user message against OpenAI chat completions."""
    response = agent.client.chat.completions.create(
        model=agent.model,
        messages=[{"role": "user", "content": question}],
        max_tokens=128,
    )
    return response.choices[0].message.content or ""


# --- AgentOpt wiring ---

def select_best_openai_agent(
    agent_factory: Callable[[str], object],
    candidate_models: Sequence[str] | Iterable[str],
    dataset_path: str = "examples/datasets/math_problems.jsonl",
):
    """Evaluate the same OpenAI chat setup across models using AgentOpt."""

    dataset = load_jsonl_dataset(dataset_path)

    # Proxy tracks the active model string; ModelSelector will mutate it.
    proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))
    runner = AgentFactoryRunner(proxy, agent_factory, run_chat)

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
        final_output = run_chat(final_agent, sample)
        print(f"Sample run with best model ({best.model_name}): {final_output}")

    return results


if __name__ == "__main__":
    select_best_openai_agent(
        agent_factory=build_client,
        candidate_models=["gpt-4o-mini", "gpt-4o"],
        dataset_path="examples/datasets/math_problems.jsonl",
    )
