# OpenAI SDK

A two-step agent (planner + solver) using the OpenAI Python SDK directly.

!!! tip "Minimal setup"
    This is the simplest integration — just pass model name strings from the `models` dict to your OpenAI client calls.

```python
from openai import OpenAI
from agentopt import BruteForceModelSelector

client = OpenAI()

class MyAgent:
    def __init__(self, models):
        self.models = models

    def run(self, input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        # Step 1: Planner generates a plan
        plan = client.chat.completions.create(
            model=self.models["planner"],
            messages=[
                {"role": "system", "content": "Create a brief plan to answer the question."},
                {"role": "user", "content": question},
            ],
        ).choices[0].message.content

        # Step 2: Solver executes the plan
        answer = client.chat.completions.create(
            model=self.models["solver"],
            messages=[
                {"role": "system", "content": f"Follow this plan and answer concisely:\n{plan}"},
                {"role": "user", "content": question},
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

selector = BruteForceModelSelector(
    agent=MyAgent,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        "solver":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)

results = selector.select_best(parallel=True)
results.print_summary()
```

[:octicons-file-code-24: Full example on GitHub](https://github.com/AgentOptimizer/agentopt/blob/main/examples/selection/local/custom_agent.py)
