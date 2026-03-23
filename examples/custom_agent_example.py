"""
Example: Custom agent (no framework) with agentopt.

This example shows how to use agentopt with a plain Python agent
that makes OpenAI SDK calls directly. No framework needed.

Prerequisites:
    1. pip install openai agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from agentopt import ModelSelector


# ---------------------------------------------------------------------------
# Step 1: Define your agent as a class with __init__(models) and run(input_data).
#
# __init__ receives a model configuration dict, e.g.
#   {"planner": "gpt-4o-mini", "solver": "gpt-4o"}
# run() takes a single datapoint and returns the agent's output.
# ---------------------------------------------------------------------------


class MyAgent:
    """A simple planner+solver agent using the OpenAI SDK."""

    def __init__(self, models):
        self.client = OpenAI()
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        # Step 1: Planner generates a plan
        plan = (
            self.client.chat.completions.create(
                model=self.planner_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a planning assistant. Create a brief plan to answer the question.",
                    },
                    {"role": "user", "content": input_data},
                ],
            )
            .choices[0]
            .message.content
        )

        # Step 2: Solver executes the plan
        answer = (
            self.client.chat.completions.create(
                model=self.solver_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Follow this plan and answer concisely:\n{plan}",
                    },
                    {"role": "user", "content": input_data},
                ],
            )
            .choices[0]
            .message.content
        )
        return answer


# ---------------------------------------------------------------------------
# Step 2: Define your evaluation dataset — (input_data, expected_output) pairs.
# We recommend 50-100 samples for production decisions,
# but even 10-20 samples can surface clear winners during development.
# ---------------------------------------------------------------------------

dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
]


# ---------------------------------------------------------------------------
# Step 3: Define your evaluation function.
# It compares agent output against expected output and returns a score.
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


# ---------------------------------------------------------------------------
# Step 4: Run model selection.
# Map each agent step to a list of candidate models.
# AgentOpt tries all combinations and ranks them by accuracy, latency, and cost.
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
