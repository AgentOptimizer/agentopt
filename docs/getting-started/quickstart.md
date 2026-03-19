# Quick Start

This guide walks you through optimizing model selection for a simple two-step agent.

## 1. Define Your Agent

Wrap your agent in a **factory function** that takes a model configuration dict and returns a callable agent:

```python
from openai import OpenAI

client = OpenAI()

def agent_maker(models):
    """Build an agent for a given model combination."""
    def run(input_data):
        # Step 1: Plan
        plan = client.chat.completions.create(
            model=models["planner"],
            messages=[{"role": "user", "content": f"Plan: {input_data}"}],
        ).choices[0].message.content

        # Step 2: Solve
        answer = client.chat.completions.create(
            model=models["solver"],
            messages=[
                {"role": "system", "content": f"Follow this plan:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer
    return run
```

The `models` dict maps each agent step (node) to a model name string. AgentOpt will try different models for each node.

## 2. Prepare Your Dataset

Create a list of `(input, expected_output)` pairs:

```python
dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
    # ... ideally ~100 samples for reliable results
]
```

## 3. Define Your Evaluation Function

The evaluation function scores agent output against the expected answer. It should return a `float` between 0 and 1 (higher is better):

```python
def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0
```

## 4. Run Model Selection

```python
from agentopt import BruteForceModelSelector

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

## 5. Use the Results

```python
# Get the best combination
best = results.get_best_combo()
print(best)  # {"planner": "gpt-4o-mini", "solver": "gpt-4.1-nano"}

# Export results
results.to_csv("results.csv")
results.export_config("optimized_config.yaml")
```

## Enabling Disk Cache

To persist cached LLM responses across runs (so re-running is instant and free):

```python
from agentopt.proxy import LLMTracker

tracker = LLMTracker(cache_dir="./llm_cache")
selector = BruteForceModelSelector(
    ...,
    tracker=tracker,
)
```

## Next Steps

- [Selection Algorithms](../concepts/algorithms.md) — Choose a smarter search strategy for large model spaces
- [How It Works](../concepts/how-it-works.md) — Understand the interception and tracking mechanism
- [Examples](../examples/openai.md) — See framework-specific examples
