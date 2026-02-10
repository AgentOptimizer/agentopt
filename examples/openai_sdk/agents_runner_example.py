"""General pattern: wrap an OpenAI Agents SDK workflow with AgentOpt.

- `build_agent(model)` constructs your normal Agents SDK Agent (e.g., Budget Helper).
- `run_agent(agent, question)` uses Runner.run_sync to execute the agent once.
- A ModelProxy holds the current model name; the invoke_fn rebuilds the agent per call
  with whatever model ModelSelector sets, so tools/logic stay untouched.
"""

from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

from openai import AssistantEventHandler  # type: ignore
from openai import OpenAI

from agentopt import ModelProxy, ModelSelector
from examples.sdk_shared import eval_fn, load_jsonl_dataset


# --- Your existing agent factory (vanilla Agents SDK) ---

def build_agent(model: str = "gpt-4o-mini"):
    """Return a plain Agents SDK Agent wired with tools; replace with real tools."""
    client = OpenAI()

    # Placeholder tool definitions; replace with actual tool implementations
    def list_expenses():
        return {
            "rent": 1200,
            "groceries": 450,
            "transport": 180,
            "entertainment": 220,
        }

    def convert_to_eur(amount_usd: float) -> float:
        return round(amount_usd * 0.92, 2)

    return client.agents.create(
        name="Budget Helper",
        model=model,
        instructions=(
            "You are a budgeting assistant. "
            "Call list_expenses to see USD amounts, then convert_to_eur for totals. "
            "Return both USD and EUR totals and highlight the top two spend categories."
        ),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "list_expenses",
                    "description": "Return monthly expenses in USD",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_to_eur",
                    "description": "Convert a USD amount to EUR",
                    "parameters": {
                        "type": "object",
                        "properties": {"amount_usd": {"type": "number"}},
                        "required": ["amount_usd"],
                    },
                },
            },
        ],
    )


def run_agent(agent, question: str) -> str:
    """Execute a single user message against an Agents SDK Agent via Runner."""
    client = OpenAI()
    runner = client.agents.runs
    result = runner.create_and_poll(agent_id=agent.id, input=question)
    if hasattr(result, "output") and result.output:
        return result.output[0].content[0].text
    return str(result)


# --- AgentOpt wiring ---

def select_best_openai_agent(
    agent_factory: Callable[[str], object],
    candidate_models: Sequence[str] | Iterable[str],
    dataset_path: str = "examples/datasets/math_problems.jsonl",
):
    """Evaluate the same Agents SDK agent across models using AgentOpt."""

    dataset = load_jsonl_dataset(dataset_path)

    # Proxy tracks the active model string; ModelSelector will mutate it.
    proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))

    def invoke_fn(payload):
        question = payload["input"]
        # Build a fresh agent with the current model
        agent = agent_factory(proxy.get_model())
        return run_agent(agent, question)

    selector = ModelSelector(
        models={proxy: list(candidate_models)},
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
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
