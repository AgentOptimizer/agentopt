# LangGraph

LangGraph graphs are compiled state machines — use `invoke_fn` instead of `agent=`.

## Example

```python
from langchain_openai import ChatOpenAI
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLM
llm = ChatOpenAI(model="gpt-4o-mini")
proxy = ModelProxy(llm)

# 2. Build your graph (proxy sits in graph nodes via closure)
graph = build_my_graph(proxy)  # your compiled StateGraph

def invoke_fn(input_data):
    return graph.invoke(input_data)

# 3. Run optimization
selector = ModelSelector(
    models={proxy: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    invoke_fn=invoke_fn,
)
results = selector.select_best(parallel=True)
```

## How parallel works

Since LangGraph graphs can't be easily deep-copied, parallel evaluation uses **thread-local model overrides**:

1. Each thread sets a per-thread model on the `ModelProxy`
2. The proxy's `_get_effective_model()` returns the thread-local model
3. The `invoke_fn` closure captures the proxy, which transparently routes to the correct model
4. After evaluation, thread-local overrides are cleaned up

This means the same graph object is shared across threads, but each thread sees its own model.
