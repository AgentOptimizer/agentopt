# AgentOpt - Optimization for AI Agents

A framework-agnostic toolkit for optimizing LLM-powered agents. Evaluate and select the best models for your agents across frameworks like CrewAI, LangChain, LlamaIndex, and the OpenAI Agents SDK — or any custom agent setup.

> **Note: This project is in early development. APIs are subject to change.**

## Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# For CrewAI support
uv sync --extra crewai

# For LlamaIndex support
uv sync --extra llamaindex
```

## Quick Start

AgentOpt works in three steps:

1. **Wrap** your LLM with `ModelProxy`
2. **Build** your agent as usual (the proxy is transparent)
3. **Run** `ModelSelector` to find the best model

### CrewAI

```python
from crewai import Agent, Task, Crew, LLM
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLM
llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))

# 2. Build your agent normally
agent = Agent(role="Researcher", goal="Answer questions", backstory="...", llm=llm)
task = Task(description="{input}", expected_output="A clear answer", agent=agent)
crew = Crew(agents=[agent], tasks=[task])

# 3. Prepare dataset: list of (input_dict, expected_answer) tuples
dataset = [
    ({"input": "What is 2 + 2?"}, "4"),
    ({"input": "Capital of France?"}, "Paris"),
]

# 4. Run optimization — auto-detects crew.kickoff()
selector = ModelSelector(
    models={llm: ["openai/gpt-4o-mini", "openai/gpt-4o"]},
    eval_fn=lambda expected, actual: expected.lower() in str(actual).lower(),
    dataset=dataset,
    agent=crew,
)
results = selector.select_best()
print(results.get_best())
```

### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLM
llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# 2. Build your agent normally
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# 3. Run optimization — auto-detects executor.invoke()
selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=executor,
)
results = selector.select_best()
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from agentopt import ModelProxy, ModelSelector

# 1. Create a real LLM (ModelProxy can't be passed directly due to Pydantic v2)
real_llm = OpenAI(model="gpt-4o-mini")

# 2. Build your agent normally
agent = FunctionAgent(
    tools=my_tools,
    llm=real_llm,
    system_prompt="You are a helpful assistant.",
)
workflow = AgentWorkflow(agents=[agent], root_agent=agent.name)

# 3. Wrap the LLM after agent creation and register
llm = ModelProxy(real_llm)

selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=workflow,  # auto-detects workflow.run() (async)
)
results = selector.select_best()
```

### OpenAI Agents SDK

```python
from agents import Agent
from agents.models.openai_provider import OpenAIProvider
from agentopt import ModelProxy, ModelSelector

# 1. Wrap an OpenAI SDK model — ModelProxy is registered as a Model ABC subclass
proxy = ModelProxy(OpenAIProvider().get_model("gpt-4o-mini"))

# 2. Build your agent with the proxy as the model
agent = Agent(name="Math QA", model=proxy, instructions="Answer math questions concisely.")

# 3. Run optimization — auto-detects via Runner.run_sync()
selector = ModelSelector(
    models={proxy: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=agent,
)
results = selector.select_best()
```

### Custom Agent / Any Framework

For agents that don't fit a supported framework, pass a custom `invoke_fn`:

```python
def my_invoke(input_data):
    """input_data is one element from your dataset tuples."""
    result = my_custom_agent.run(input_data["input"])
    return result

selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    invoke_fn=my_invoke,  # use this instead of agent=
)
```

## Core Concepts

### Adapter Architecture

This is the internal design that makes framework auto-detection work. You don't need to understand it to use AgentOpt, but it's useful to know if you're debugging or adding a new framework.

**The problem it solves:** Without adapters, every time we added a new framework (CrewAI, LangChain, LlamaIndex, etc.) we had to edit three separate files — the proxy, the selector's `__init__`, and the parallel cloning logic. All three had the same `if is_crewai: ... elif is_langchain: ...` chains. Adding framework #5 meant touching all three.

**The solution:** One `FrameworkAdapter` class per framework. Each adapter answers four questions about its framework:

```
1. detect()            — Is this agent object mine?
2. get_invoke_fn()     — How do I call the agent to get a result?
3. register_with_proxy() — How do I wire sync callbacks so set_model() propagates?
4. clone_for_parallel()  — How do I make a deep-safe independent copy for a thread?
```

**How it fits together at runtime:**

```
User calls: ModelSelector(agent=my_crew, ...)
                │
                ▼
         get_adapter(my_crew)
                │  iterates _REGISTRY, calls each adapter.detect()
                │  CrewAIAdapter.detect() → type(crew).__module__.startswith("crewai") → True ✓
                ▼
         CrewAIAdapter
          ├── get_invoke_fn()     → returns crew.kickoff (bound method)
          ├── register_with_proxy() → adds sync closures to proxy._sync_callbacks
          └── clone_for_parallel() → model_copy(deep=False) + clone_crew_agents()
```

**Self-registration:** Each framework file registers its own adapter at import time — no central list to maintain:

```python
# bottom of crewai.py
register_adapter(CrewAIAdapter())   # runs when crewai.py is first imported

# bottom of langchain.py
register_adapter(LangChainAdapter())
# ... etc.
```

These imports happen automatically when `agentopt` is loaded (via the `model_proxy/__init__.py` import chain), so adapters are always ready before any user code runs.

**Adding a new framework** is just one new file:

```python
# model_proxy/myframework.py
class MyFrameworkAdapter(FrameworkAdapter):
    invoke_method_name = "run"

    def detect(self, agent):
        return type(agent).__module__.startswith("myframework")

    def get_invoke_fn(self, agent):
        return agent.run

    def register_with_proxy(self, proxy, agent, all_proxies):
        def _sync(new_llm):
            agent.llm = new_llm          # however your framework swaps models
        proxy._add_sync(_sync)

    def clone_for_parallel(self, agent, proxies, combo, get_model_name):
        fresh_llm = build_my_llm(get_model_name(combo[0]))
        return agent.copy(llm=fresh_llm)  # however your framework clones

register_adapter(MyFrameworkAdapter())
```

No other files need to change.

### ModelProxy

A transparent wrapper around any LLM object that enables model swapping without rebuilding agents. It forwards all attribute access and method calls to the underlying model.

```python
from agentopt import ModelProxy

proxy = ModelProxy(original_llm)

# Behaves exactly like original_llm
proxy.invoke(...)   # forwarded
proxy.temperature   # forwarded

# Swap the underlying model (by string or full object)
proxy.set_model("gpt-4o")           # rebuilds the model with the new name
proxy.set_model(new_llm_instance)   # replaces entirely

# Register an agent for automatic sync on every set_model() call
proxy.register(my_agent)   # auto-detects framework

# Inspect
proxy.get_model()  # returns the current underlying model
```

**Why is this needed?** Agent frameworks capture the model reference at construction time. Without `ModelProxy`, you'd need to rebuild the entire agent pipeline for every model you want to test. The proxy provides a stable reference that agents hold, while the actual model behind it can be swapped.

```
Agent --> ModelProxy --> LLM (gpt-4o-mini)
                    --> LLM (gpt-4o)        # swapped, agent unchanged
```

**String-based swapping:** When you pass a string to `set_model()`, the proxy rebuilds the correct LLM class (`ChatOpenAI`, `ChatAnthropic`, `crewai.LLM`, etc.) based on the model name prefix. Cross-provider swaps work automatically.

### ModelSelector

Evaluates your agent across model combinations and selects the best one based on accuracy and latency.

```python
from agentopt import ModelSelector

selector = ModelSelector(
    models=models,       # {ModelProxy: [candidate_models]}
    eval_fn=eval_fn,     # (expected, actual) -> bool | float
    dataset=dataset,     # [(input_data, expected_answer), ...]
    agent=agent,         # or invoke_fn=callable
)
results = selector.select_best()

# Parallel evaluation (runs all combinations concurrently)
results = selector.select_best(parallel=True)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | `dict[ModelProxy, list]` | Maps each proxy to its candidate models (strings or objects) |
| `eval_fn` | `(str, Any) -> bool \| float` | Compares expected answer to actual output. Returns bool or float score (higher is better) |
| `dataset` | `list[tuple[Any, str]]` | List of `(input_data, expected_answer)` tuples. `input_data` is passed directly to the agent's invoke method |
| `agent` | `Any` | Agent object (CrewAI Crew, LangChain AgentExecutor, LlamaIndex AgentWorkflow, OpenAI Agents SDK Agent). Mutually exclusive with `invoke_fn` |
| `invoke_fn` | `callable` | Custom function `(input_data) -> result`. Mutually exclusive with `agent` |

### Multi-Agent / Multi-LLM Optimization

When your pipeline uses multiple LLMs (e.g., different agents with different models), create a separate `ModelProxy` for each and pass them all to `ModelSelector`. It evaluates the **Cartesian product** of all candidate combinations.

```python
researcher_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
coder_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# ... build agents with these proxies ...

# Register each proxy with its executor for sync on set_model()
researcher_llm.register(researcher_executor)
coder_llm.register(coder_executor)

selector = ModelSelector(
    models={
        researcher_llm: ["gpt-4o-mini", "gpt-4o"],
        coder_llm: ["gpt-4o-mini", "gpt-4o"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
    invoke_fn=my_pipeline,  # your function that runs both agents
)
# Tests all 4 combinations: (mini,mini), (mini,4o), (4o,mini), (4o,4o)
results = selector.select_best()
```

### SelectionResults

```python
results = selector.select_best()

results.get_best()              # -> ModelResult (best overall)
results.get_best("combination") # -> ModelResult (best for a specific attribute)

for r in results:
    print(f"{r.model_name}: acc={r.accuracy:.2%}, latency={r.latency_seconds:.1f}s")

results.to_csv("results.csv")  # export to CSV
```

Each `ModelResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Name of the model or combination |
| `accuracy` | `float` | Average score across the dataset |
| `latency_seconds` | `float` | Average latency per evaluation |
| `attribute` | `str` | Grouping label (e.g., `"combination"`) |
| `is_best` | `bool` | Whether this was the best result |

## Dataset Format

The dataset is a list of `(input_data, expected_answer)` tuples that you prepare yourself. `input_data` can be any type your agent's invoke method accepts (typically a dict). `expected_answer` is a string passed to your `eval_fn`.

A common pattern is loading from JSONL:

```python
import json

def load_dataset(path):
    tasks = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks
```

JSONL format:
```jsonl
{"question": "What is 15 + 27?", "output": "42"}
{"question": "What is the capital of France?", "output": "Paris"}
```

## Project Structure

```
agentopt/
├── src/agentopt/
│   ├── __init__.py              # Public API exports
│   ├── base_models.py           # Type aliases (EvalFn, ModelSpec, ModelsConfig)
│   ├── model_factory.py         # create_model_from_string — multi-provider LLM factory
│   ├── model_topology.py        # Model quality/speed rankings for hill climbing
│   └── model_proxy/
│       ├── base.py              # ModelProxy — transparent proxy, set_model(), register()
│       ├── adapter.py           # FrameworkAdapter ABC + registry (get_adapter, register_adapter)
│       ├── constants.py         # Framework detection helpers + MODEL_FIELDS
│       ├── builders.py          # Generic LLM rebuild helpers
│       ├── crewai.py            # CrewAI support + CrewAIAdapter
│       ├── langchain.py         # LangChain support + LangChainAdapter + extract_prompt
│       ├── llamaindex.py        # LlamaIndex support + LlamaIndexAdapter
│       └── openai_sdk.py        # OpenAI Agents SDK support + OpenAISDKAdapter
└── model_selection/
    ├── base.py                  # BaseModelSelector, ModelResult, SelectionResults
    ├── brute_force.py           # BruteForceModelSelector (default ModelSelector)
    ├── hill_climbing.py         # HillClimbingModelSelector (experimental)
    └── utils.py                 # Compat re-export of extract_prompt

examples/
├── crewai_example.py            # CrewAI: single-agent, multi-agent, parallel
├── langchain_example.py         # LangChain: single-agent, multi-agent with chaining
├── llamaindex_example.py        # LlamaIndex: single, multi-agent, multi-LLM
├── openai_sdk_example.py        # OpenAI Agents SDK
└── datasets/
    └── math_problems.jsonl
```

## Environment Setup

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here   # for claude-* models
export GOOGLE_API_KEY=your_key_here      # for gemini-* models
```

Or use a `.env` file:
```bash
cp .env.example .env
```

AgentOpt also supports OpenRouter as a universal fallback:

```bash
export OPENROUTER_API_KEY=your_key_here
```

## API Reference

### Exports

```python
from agentopt import (
    # Core
    ModelProxy,              # Transparent LLM proxy with auto-framework-detection
    ModelSelector,           # Brute-force model selector (default)
    BaseModelSelector,       # Abstract base for custom selectors

    # Results
    ModelResult,             # Single model evaluation result
    SelectionResults,        # Container for all results

    # Utilities
    create_model_from_string,  # Create LangChain model from string name
    normalize_models,          # Normalize model config dicts

    # Type aliases
    EvalFn,                  # Callable[[str, Any], bool | float]
    ModelSpec,               # str | Any
    ModelsConfig,            # dict[ModelProxy, list[ModelSpec]]
)
```

## License

MIT
