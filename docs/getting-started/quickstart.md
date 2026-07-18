# Quick Start

Optimize model selection for a multi-step agent in under 5 minutes.

## Overview

AgentOpt finds the best LLM model combination for your agent. Here's the idea:

1. You define an **agent class** — it accepts a model config and runs on a datapoint
2. You provide **candidate models** for each step of your agent
3. AgentOpt tries different combinations, scores them, and reports the best ones

```
┌─────────────────────────────────────────────────────────────────┐
│  models = {                                                     │
│      "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],    │
│      "solver":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],    │
│  }                                                              │
│                                                                 │
│  For each combination (e.g. planner=gpt-4o-mini, solver=gpt-4o)│
│  AgentOpt does:                                                 │
│                                                                 │
│    agent = MyAgent({"planner": "gpt-4o-mini", "solver": "gpt-4o"})
│                                                                 │
│    for input_data, expected in dataset:                         │
│        output = agent.run(input_data)                           │
│        score  = eval_fn(expected, output)                       │
│                                                                 │
│  Then ranks all combinations by accuracy / cost / latency.      │
└─────────────────────────────────────────────────────────────────┘
```

Smart selection algorithms skip combinations that are clearly worse, and parallelization makes everything fast. Response caching ensures identical LLM calls are never repeated.

---

## Step 1: Define Your Agent

Write a class with two methods:

- `__init__(self, models)` — receives a `dict` mapping step names to model names (e.g. `{"planner": "gpt-4o-mini", "solver": "gpt-4o"}`)
- `run(self, input_data)` — processes one datapoint and returns the output

```python
from openai import OpenAI

class MyAgent:
    def __init__(self, models):
        self.client = OpenAI()
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        # Step 1: Plan
        plan = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[{"role": "user", "content": f"Plan: {input_data}"}],
        ).choices[0].message.content

        # Step 2: Solve
        answer = self.client.chat.completions.create(
            model=self.solver_model,
            messages=[
                {"role": "system", "content": f"Follow this plan:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer
```

!!! info "No base class required"
    Just implement `__init__` and `run` — duck typing only. Works with any framework: OpenAI, LangChain, CrewAI, LlamaIndex, etc.

## Step 2: Prepare Your Dataset

Create a list of `(input, expected_output)` tuples:

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

!!! tip "Dataset size"
    More samples means more reliable rankings. We recommend **50-100 samples** for production decisions, but even 10-20 samples can surface clear winners during development.

## Step 3: Define Your Evaluation Function

Score agent output against the expected answer. Return a `float` in `[0, 1]`:

```python
def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0
```

The `eval_fn` is called on the output of `agent.run(input_data)` for each datapoint.

## Step 4: Run Model Selection

```python
from agentopt import BruteForceModelSelector

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

The `models` dict maps each step name (matching the keys your `__init__` expects) to a **list of candidates**. AgentOpt picks one from each list, constructs `MyAgent({"planner": pick1, "solver": pick2})`, and evaluates it across your dataset.

With 3 candidates per step and 2 steps, that's 9 combinations. Smart algorithms like `ArmEliminationModelSelector` or `BayesianOptimizationModelSelector` can find the best combination without evaluating all of them — and they also select which datapoints to run on, stopping early when the winner is clear.

!!! tip "Optional: trade accuracy for cost or latency"
    Add `lambda_cost=0.3` and/or `lambda_latency=0.2` to the selector constructor to rank combinations by a scalar combined objective instead of accuracy alone. Omit them (default `0.0`) for accuracy-first selection. Details: [Combined objective](../api/selectors.md#combined-objective-optional-costlatency-weights).

## Step 5: Use the Results

```python
# Get the winning combination
best = results.get_best_combo()
print(best)  # {"planner": "gpt-4o-mini", "solver": "gpt-4.1-nano"}

# Export for later use
results.to_csv("results.csv")
results.export_config("optimized_config.yaml")
```

## Enable Disk Cache

Persist cached responses across runs so re-running is instant and free:

```python
from agentopt.proxy import LLMTracker

tracker = LLMTracker(cache_dir="./llm_cache")
selector = BruteForceModelSelector(
    ...,
    tracker=tracker,
)
```

!!! success "Cache survives restarts"
    With disk caching enabled, if a run is interrupted or you tweak your eval function, all previously-seen LLM calls are served from cache. No API cost, no latency.

---

**Next steps:**

- [Selection Algorithms](../concepts/algorithms.md) — Choose a smarter strategy for large model spaces
- [How It Works](../concepts/how-it-works.md) — Understand the interception mechanism
- [Examples](../examples/openai.md) — Framework-specific examples
