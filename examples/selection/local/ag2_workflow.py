"""
Example: AG2 agent with agentopt.

Prerequisites:
    1. pip install "ag2[openai]>=1,<2" "agentopt-py[plot]"
    2. Set OPENAI_API_KEY environment variable
"""

from ag2 import Agent
from ag2.config import OpenAICompletionsConfig
from dotenv import load_dotenv

from agentopt import ModelSelector

load_dotenv()


# ---------------------------------------------------------------------------
# Step 1: Define your agent class.
# __init__(models) receives a dict like {"planner": "gpt-4o", "solver": "gpt-4o-mini"}.
# run(input_data) runs the agent on a single datapoint and returns the output.
# ---------------------------------------------------------------------------


class MyAgent:
    """AG2 planner+solver agent pair."""

    def __init__(self, models):
        self.planner = Agent(
            "Planner",
            prompt="Create a brief plan for answering the question.",
            config=OpenAICompletionsConfig(model=models["planner"]),
        )
        self.solver = Agent(
            "Solver",
            prompt="Use the plan to produce a concise final answer.",
            config=OpenAICompletionsConfig(model=models["solver"]),
        )

    async def run(self, input_data):
        plan = await self.planner.ask(input_data)
        answer = await self.solver.ask(f"Question: {input_data}\nPlan: {plan.body}")
        return answer.body


# ---------------------------------------------------------------------------
# Step 2: Evaluation dataset — (input_data, expected_output) pairs.
# ---------------------------------------------------------------------------

dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
]


# ---------------------------------------------------------------------------
# Step 3: Evaluation function — score agent output against expected answer.
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


# ---------------------------------------------------------------------------
# Step 4: Run model selection.
# Two steps ("planner", "solver") × 3 models = 9 combinations.
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
        method="auto",  # Gittins selection
    )

    results = selector.select_best(parallel=True)
    results.print_summary()
    results.plot_pareto()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
