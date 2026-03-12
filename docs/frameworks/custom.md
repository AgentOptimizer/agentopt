# Custom Agents

For agents that don't fit a supported framework, pass a custom `invoke_fn`.

## Example

```python
from agentopt import ModelProxy, ModelSelector
from langchain_openai import ChatOpenAI

llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

def my_invoke(input_data):
    """input_data is one element from your dataset tuples."""
    response = llm.invoke(input_data["input"])
    return response.content

selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    invoke_fn=my_invoke,
)
results = selector.select_best(parallel=True)
```

## How parallel works with invoke_fn

When `parallel=True` and you use `invoke_fn`:

1. The selector creates a fresh LLM instance per thread
2. It sets a **thread-local model override** on the `ModelProxy`
3. Your `invoke_fn` closure captures the proxy, which transparently routes to the thread's model
4. After evaluation, overrides are cleaned up

This means your `invoke_fn` code doesn't need any threading logic — it works exactly the same in sequential and parallel modes.

## Multi-LLM custom pipelines

```python
planner = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
executor = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

def my_pipeline(input_data):
    plan = planner.invoke(f"Plan: {input_data['input']}")
    result = executor.invoke(f"Execute: {plan.content}")
    return result.content

selector = ModelSelector(
    models={
        planner: ["gpt-4o-mini", "gpt-4o"],
        executor: ["gpt-4o-mini", "gpt-4o"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
    invoke_fn=my_pipeline,
)
# Tests all 4 combinations
results = selector.select_best(parallel=True)
```
