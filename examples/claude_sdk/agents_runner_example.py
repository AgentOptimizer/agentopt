"""General pattern: wrap a Claude Messages workflow with AgentOpt.

- `build_agent(model)` is your normal Claude Messages setup (tools optional).
- `run_agent(agent_cfg, question)` sends one message via Anthropic and returns text.
- ModelProxy holds the current model name; the invoke_fn rebuilds the agent config
  with whatever model ModelSelector sets, so you keep the same prompt/tools logic.
"""

from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

from anthropic import Anthropic

from agentopt import ModelProxy, ModelSelector
from examples.sdk_shared import eval_fn, load_jsonl_dataset


# --- Your existing agent "factory" (Claude Messages config) ---

def build_agent(model: str = "claude-3-5-haiku-latest"):
    """Return a simple Claude Messages config; replace prompt/tools as needed."""
    return SimpleNamespace(model=model)


def run_agent(agent_cfg, question: str) -> str:
    """Execute a single user message against Claude Messages API."""
    client = Anthropic()
    message = client.messages.create(
        model=agent_cfg.model,
        max_tokens=256,
        messages=[{"role": "user", "content": question}],
    )
    if message.content and hasattr(message.content[0], "text"):
        return message.content[0].text
    return str(message)


# --- AgentOpt wiring ---

def select_best_claude_agent(
    agent_factory: Callable[[str], object],
    candidate_models: Sequence[str] | Iterable[str],
    dataset_path: str = "examples/datasets/math_problems.jsonl",
):
    """Evaluate the same Claude Messages setup across models using AgentOpt."""

    dataset = load_jsonl_dataset(dataset_path)

    proxy = ModelProxy(SimpleNamespace(model="claude-3-5-haiku-latest"))

    def invoke_fn(payload):
        question = payload["input"]
        agent_cfg = agent_factory(proxy.get_model())
        return run_agent(agent_cfg, question)

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
    )

    results = selector.select_best()
    best = results.get_best()
    print(f"Best Claude model: {best}")

    if best:
        proxy.set_model(best.model_name)
        final_agent = agent_factory(proxy.get_model())
        sample = dataset[0][0]["input"]
        final_output = run_agent(final_agent, sample)
        print(f"Sample run with best model ({best.model_name}): {final_output}")

    return results


if __name__ == "__main__":
    select_best_claude_agent(
        agent_factory=build_agent,
        candidate_models=["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
        dataset_path="examples/datasets/math_problems.jsonl",
    )
