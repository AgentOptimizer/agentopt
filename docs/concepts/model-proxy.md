# ModelProxy

`ModelProxy` is a transparent wrapper around any LLM object that enables model swapping without rebuilding agents.

## Why is it needed?

Agent frameworks capture the model reference at construction time. Without `ModelProxy`, you'd need to rebuild the entire agent pipeline for every model you want to test. The proxy provides a stable reference that agents hold, while the actual model behind it can be swapped.

## Basic usage

```python
from langchain_openai import ChatOpenAI
from agentopt import ModelProxy

proxy = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# Behaves exactly like the original LLM
response = proxy.invoke("Hello!")
print(proxy.model_name)  # forwarded attribute access
```

## Swapping models

```python
# Swap by string — rebuilds the correct LLM class automatically
proxy.set_model("gpt-4o")

# Swap by object — replaces the underlying model entirely
proxy.set_model(ChatOpenAI(model="gpt-4o", temperature=0.5))

# Inspect the current model
current = proxy.get_model()
```

**String-based swapping** detects the provider from the model name prefix (`openai/`, `anthropic/`, `google/`, etc.) and builds the appropriate LLM class. Cross-provider swaps work automatically.

## Registering agents

When an agent is registered with a proxy, `set_model()` automatically propagates the new model to the agent:

```python
proxy = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
agent = AgentExecutor(agent=my_agent, tools=tools)

# Auto-detects framework and wires sync callbacks
proxy.register(agent)

# Now set_model() also updates the agent's internal LLM reference
proxy.set_model("gpt-4o")
```

!!! tip
    You don't need to call `register()` manually when using `ModelSelector` — the selector handles registration automatically.

## Multi-agent pipelines

Create a separate proxy for each LLM in your pipeline:

```python
researcher_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
writer_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# Build agents with different proxies
researcher = Agent(llm=researcher_llm, ...)
writer = Agent(llm=writer_llm, ...)

# Optimize both independently
selector = ModelSelector(
    models={
        researcher_llm: ["gpt-4o-mini", "gpt-4o"],
        writer_llm: ["gpt-4o-mini", "gpt-4o"],
    },
    ...
)
# Tests all 4 combinations: (mini,mini), (mini,4o), (4o,mini), (4o,4o)
```

## Thread safety

`ModelProxy` supports thread-local model overrides for parallel evaluation. Each thread sees its own model instance — no shared state, no conflicts. This is handled automatically by the selector when `parallel=True`.
