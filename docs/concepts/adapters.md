# Framework Adapters

Adapters are the internal mechanism that makes framework auto-detection work. You don't need to understand adapters to use AgentOpt, but they're useful to know when debugging or adding support for a new framework.

## The problem

Without adapters, every time a new framework was added, three separate files needed editing — the proxy, the selector's `__init__`, and the parallel cloning logic. All three had the same `if is_crewai: ... elif is_langchain: ...` chains.

## The solution

One `FrameworkAdapter` class per framework. Each adapter answers four questions:

```python
class FrameworkAdapter:
    def detect(self, agent)              # Is this agent object mine?
    def get_invoke_fn(self, agent)       # How do I call the agent?
    def register_with_proxy(self, ...)   # How do I wire model sync?
    def clone_for_parallel(self, ...)    # How do I make thread-safe copies?
```

## How it works at runtime

```
User calls: ModelSelector(agent=my_crew, ...)
                |
                v
         get_adapter(my_crew)
                |  iterates registry, calls each adapter.detect()
                |  CrewAIAdapter.detect() → True
                v
         CrewAIAdapter
          +-- get_invoke_fn()       → returns crew.kickoff
          +-- register_with_proxy() → adds sync closures
          +-- clone_for_parallel()  → deep-copies with fresh LLMs
```

Each framework file registers its own adapter at import time:

```python
# Bottom of crewai.py
register_adapter(CrewAIAdapter())
```

## Adding a new framework

Create a single file in `model_proxy/framework_specific_implementation/`:

```python
from ..adapter import FrameworkAdapter, register_adapter

class MyFrameworkAdapter(FrameworkAdapter):
    invoke_method_name = "run"

    def detect(self, agent):
        return type(agent).__module__.startswith("myframework")

    def get_invoke_fn(self, agent):
        return agent.run

    def register_with_proxy(self, proxy, agent, all_proxies):
        def _sync(new_llm):
            agent.llm = new_llm
        proxy._add_sync(_sync)

    def clone_for_parallel(self, agent, proxies, combo, get_model_name):
        fresh_llm = build_my_llm(get_model_name(combo[0]))
        return agent.copy(llm=fresh_llm)

register_adapter(MyFrameworkAdapter())
```

No other files need to change.

## Supported adapters

| Framework | Adapter | Invoke method | Parallel support |
|-----------|---------|---------------|------------------|
| CrewAI | `CrewAIAdapter` | `.kickoff()` | Clone-based |
| LangChain | `LangChainAdapter` | `.invoke()` | Clone-based |
| LlamaIndex | `LlamaIndexAdapter` | `.run()` (async) | Clone-based |
| OpenAI SDK | `OpenAISDKAdapter` | `Runner.run_sync()` | Clone or thread-local |
| AG2 | `AG2Adapter` | `.run()` | Clone or thread-local |
| LangGraph | N/A (uses `invoke_fn`) | — | Thread-local |
| Claude SDK | N/A (uses `invoke_fn`) | — | Thread-local |
