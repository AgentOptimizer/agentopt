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


class MyAgent:
    """A simple planner+solver agent using the OpenAI SDK."""

    def __init__(self, models):
        self.client = OpenAI()
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        # Step 1: Planner generates a plan
        plan = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[
                {"role": "system", "content": "You are a planning assistant. Create a brief plan to answer the question."},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content

        # Step 2: Solver executes the plan
        answer = self.client.chat.completions.create(
            model=self.solver_model,
            messages=[
                {"role": "system", "content": f"Follow this plan and answer concisely:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
]


if __name__ == "__main__":
    selector = ModelSelector(
        agent=MyAgent,
        models={
            "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
            "solver": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        },
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",
    )

    results = selector.select_best(parallel=True)
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
