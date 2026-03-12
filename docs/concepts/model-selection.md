# Model Selection

Model selection evaluates your agent across candidate model combinations and picks the best one based on accuracy and latency.

## Core interface

```python
from agentopt import ModelSelector  # alias for BruteForceModelSelector

selector = ModelSelector(
    models=models,       # {ModelProxy: [candidate_models]}
    eval_fn=eval_fn,     # (expected, actual) -> bool | float
    dataset=dataset,     # [(input_data, expected_answer), ...]
    agent=agent,         # or invoke_fn=callable
)
results = selector.select_best()
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | `dict[ModelProxy, list]` | Maps each proxy to its candidate models (strings or objects) |
| `eval_fn` | `(str, Any) -> bool \| float` | Compares expected answer to actual output |
| `dataset` | `list[tuple[Any, str]]` | List of `(input_data, expected_answer)` tuples |
| `agent` | `Any` | Agent object — framework is auto-detected. Mutually exclusive with `invoke_fn` |
| `invoke_fn` | `callable` | Custom `(input_data) -> result` function. Mutually exclusive with `agent` |

## Agent vs invoke_fn

Use **`agent=`** when your framework has a standard invoke method that AgentOpt can auto-detect (CrewAI, LangChain, LlamaIndex, OpenAI SDK, AG2):

```python
selector = ModelSelector(
    models={llm: candidates},
    eval_fn=eval_fn,
    dataset=dataset,
    agent=crew,  # auto-detects crew.kickoff()
)
```

Use **`invoke_fn=`** for LangGraph, Claude SDK, custom pipelines, or any case where you need control over how the agent is called:

```python
def my_invoke(input_data):
    result = graph.invoke(input_data)
    return result["output"]

selector = ModelSelector(
    models={llm: candidates},
    eval_fn=eval_fn,
    dataset=dataset,
    invoke_fn=my_invoke,
)
```

## Parallel evaluation

Pass `parallel=True` to evaluate all combinations concurrently:

```python
results = selector.select_best(parallel=True, max_workers=4)
```

**With `agent=`:** The selector clones the agent per combination via framework-specific adapters, each clone getting its own fresh LLM instance.

**With `invoke_fn=`:** Each thread sets per-thread model overrides on the `ModelProxy` via thread-local storage. The `invoke_fn` closure captures the proxy, which transparently routes to the thread's model.

## SelectionResults

```python
results = selector.select_best()

# Best overall
best = results.get_best()
print(f"{best.model_name}: {best.accuracy:.1%}, {best.latency_seconds:.1f}s")

# Iterate all results
for r in results:
    print(r)

# Print formatted table
results.print_summary()

# Export
results.to_csv("results.csv")
```

Each `ModelResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Name of the model or combination |
| `accuracy` | `float` | Average score across the dataset |
| `latency_seconds` | `float` | Average latency per evaluation |
| `input_tokens` | `dict[str, int]` | Input tokens per model |
| `output_tokens` | `dict[str, int]` | Output tokens per model |
| `is_best` | `bool` | Whether this was the winning result |

## Selection criteria

The selector picks the best combination using:

1. **Highest accuracy** wins
2. **Ties broken by lowest latency**

This ensures you get the most accurate model, and among equally accurate models, the fastest one.
