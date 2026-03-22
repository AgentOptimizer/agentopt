# Quick Start

Optimize model selection for a two-step agent in under 5 minutes.

## Step 1: Define Your Agent

Define your agent as a **class** with `__init__(self, models)` and `run(self, input_data)` methods:

```python
from openai import OpenAI

client = OpenAI()

class MyAgent:
    """An agent for a given model combination."""
    def __init__(self, models):
        self.models = models

    def run(self, input_data):
        # Step 1: Plan
        plan = client.chat.completions.create(
            model=self.models["planner"],
            messages=[{"role": "user", "content": f"Plan: {input_data}"}],
        ).choices[0].message.content

        # Step 2: Solve
        answer = client.chat.completions.create(
            model=self.models["solver"],
            messages=[
                {"role": "system", "content": f"Follow this plan:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer
```

!!! info "How it works"
    The `models` dict maps each agent step (node) to a model name string. AgentOpt will swap in different models for each node and measure the results. No base class is required — just implement `__init__(self, models)` and `run(self, input_data)`.

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
