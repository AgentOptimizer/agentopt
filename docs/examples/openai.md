# OpenAI SDK Example

A simple two-step agent (planner + solver) using the OpenAI SDK directly.

```python
from openai import OpenAI
from agentopt import BruteForceModelSelector

client = OpenAI()

def agent_maker(models):
    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        # Step 1: Planner generates a plan
        plan = client.chat.completions.create(
            model=models["planner"],
            messages=[
                {"role": "system", "content": "Create a brief plan to answer the question."},
                {"role": "user", "content": question},
            ],
        ).choices[0].message.content

        # Step 2: Solver executes the plan
        answer = client.chat.completions.create(
            model=models["solver"],
            messages=[
                {"role": "system", "content": f"Follow this plan and answer concisely:\n{plan}"},
                {"role": "user", "content": question},
            ],
        ).choices[0].message.content
        return answer
    return run

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
    agent_fn=agent_maker,
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

See full example: [examples/custom_agent_example.py](https://github.com/AgentOptimizer/agentopt/blob/main/examples/custom_agent_example.py)
