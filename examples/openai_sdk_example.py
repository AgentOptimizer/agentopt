"""
Example: OpenAI Agents SDK agent with agentopt.

Prerequisites:
    1. pip install openai-agents agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, function_tool

from agentopt import ModelSelector


@function_tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"Search results for: {query}"


# ---------------------------------------------------------------------------
# Step 1: Define your agent class.
# __init__(models) receives a dict like {"planner": "gpt-4o", "solver": "gpt-4o-mini"}.
# run(input_data) runs the agent on a single datapoint and returns the output.
# ---------------------------------------------------------------------------


class MyAgent:
    """OpenAI Agents SDK planner+solver agent pair."""

    def __init__(self, models):
        planner = Agent(
            name="Planner",
            model=models["planner"],
            instructions="Create a brief plan to answer the question. Be concise.",
            tools=[search],
        )
        solver = Agent(
            name="Solver",
            model=models["solver"],
            instructions="Given a plan, produce a concise final answer.",
            handoffs=[],
        )
        self.planner = planner.clone(handoffs=[solver])

    def run(self, input_data):
        result = Runner.run_sync(self.planner, input_data)
        return result.final_output


# ---------------------------------------------------------------------------
# Step 2: Evaluation dataset — (input_data, expected_output) pairs.
# ---------------------------------------------------------------------------

dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
]


# ---------------------------------------------------------------------------
# Step 3: Evaluation function — score agent output against expected answer.
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


# ---------------------------------------------------------------------------
# Step 4: Run model selection.
# Map each agent step to candidate models. AgentOpt evaluates all combinations.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    selector = ModelSelector(
        agent=MyAgent,
        models={
            "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
            "solver": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        },
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",  # or "auto" for smarter selection algorithms
    )

    results = selector.select_best(parallel=True)
    results.print_summary()
    results.plot_pareto()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
