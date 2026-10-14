# Quick Start

Set up model selection for a multi-step agent using your own evaluation examples.

## Overview

AgentOpt evaluates candidate LLM combinations and recommends a configuration
for your agent. Here's the idea:

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

Adaptive selectors decide which evaluations to run next based on the observations
collected so far. Parallel evaluation can reduce waiting time, and response
caching can reuse matching requests once a response is available. See
[Run Your Benchmark](your-benchmark.md) for live recommendations and stopping.

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
from agentopt import ModelSelector

selector = ModelSelector(
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

With 3 candidates per step and 2 steps, that's 9 combinations. The default `ModelSelector` uses Gittins to allocate further evaluations after a warm start on each combination. Set `observation_budget_fraction` to limit configuration–question evaluations, optionally add `max_search_cost_usd` to limit spending, or stop while watching live recommendations. Single mode advises when to stop but continues by default; use `stop_on_recommendation=True` for automatic stopping. Use `objective_mode="multi"` to inspect accuracy/cost alternatives.

!!! tip "Optional: trade accuracy for cost or latency"
    Add `lambda_cost=0.3` and/or `lambda_latency=0.2` to the selector constructor to rank combinations by a scalar combined objective instead of accuracy alone. Omit them (default `0.0`) for accuracy-first selection. Details: [Combined objective](../api/selectors.md#combined-objective-optional-costlatency-weights).

## Step 5: Use the Results

```python
# Get the winning combination
best = results.get_best_combo()
print(best)  # {"planner": "gpt-4o-mini", "solver": "gpt-4.1-nano"}

# Save the result, even if an early stop leaves no eligible recommendation
from pathlib import Path
Path("results.json").write_text(results.model_dump_json(indent=2), encoding="utf-8")
if best is not None:
    results.export_config("optimized_config.yaml")
```

## Reuse Cached Responses

In local mode, caching is enabled by default in `.agentopt_cache`. Choose a
directory to keep responses for repeated runs of this benchmark:

```python
from agentopt.proxy import LLMTracker

tracker = LLMTracker(cache_dir="./llm_cache")
selector = ModelSelector(
    ...,
    tracker=tracker,
)
```

!!! info "Re-running after a stop"
    Create a new selector to start another run. Identical requests already saved in the cache can reuse responses without another provider call. New or changed requests can still incur charges, and scoring and selection still take time. Stopping does not save a resumable selection checkpoint; disk caching preserves responses, not the selector's progress. A forced process exit can lose entries not yet flushed to disk. See [Response Caching](../concepts/caching.md).

---

**Next steps:**

- [Run Your Own Benchmark](your-benchmark.md) — Load examples, watch recommendations, stop a run, and choose nondefault settings
- [Selection Algorithms](../concepts/algorithms.md) — Choose a smarter strategy for large model spaces
- [How It Works](../concepts/how-it-works.md) — Understand the interception mechanism
- [Examples](../examples/openai.md) — Framework-specific examples
