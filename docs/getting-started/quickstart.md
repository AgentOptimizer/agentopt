# Quick Start

This guide walks through a minimal AgentOpt workflow.

## 1. Prepare your dataset

A dataset is a list of `(input_data, expected_answer)` tuples:

```python
dataset = [
    ({"input": "What is 2 + 2?"}, "4"),
    ({"input": "Capital of France?"}, "Paris"),
    ({"input": "Largest planet in our solar system?"}, "Jupiter"),
]
```

You can also load from JSONL:

```python
import json

def load_dataset(path):
    tasks = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            tasks.append(({"input": item["question"]}, item["answer"]))
    return tasks

dataset = load_dataset("my_data.jsonl")
```

## 2. Define an evaluation function

The eval function compares expected vs actual output and returns a score:

```python
# Simple substring match
def eval_fn(expected, actual):
    return expected.lower() in str(actual).lower()

# Or return a float score (0.0 to 1.0)
def eval_fn(expected, actual):
    if expected.lower() == str(actual).strip().lower():
        return 1.0
    elif expected.lower() in str(actual).lower():
        return 0.5
    return 0.0
```

## 3. Wrap your LLM and run selection

```python
from langchain_openai import ChatOpenAI
from agentopt import ModelProxy, ModelSelector

# Wrap the LLM
llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# Define how to call the model
def invoke_fn(input_data):
    response = llm.invoke(input_data["input"])
    return response.content

# Run model selection
selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=eval_fn,
    dataset=dataset,
    invoke_fn=invoke_fn,
)
results = selector.select_best()
results.print_summary()
```

## 4. Use the results

```python
# Get the best model
best = results.get_best()
print(f"Best: {best.model_name} ({best.accuracy:.1%} accuracy, {best.latency_seconds:.1f}s)")

# The proxy is already set to the best model
response = llm.invoke("What is 3 + 5?")

# Export to CSV
results.to_csv("results.csv")
```

## Next steps

- Learn about [ModelProxy](../concepts/model-proxy.md) in depth
- Explore [selection strategies](../concepts/strategies.md) beyond brute force
- See [framework-specific guides](../frameworks/overview.md) for your agent framework
