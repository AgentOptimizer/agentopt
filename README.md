# AgentOpt - Optimization for AI Agents

A framework-agnostic toolkit for optimizing LLM-powered agents. Evaluate and select the best models for your agents across frameworks like CrewAI, LangChain, LangGraph, and LlamaIndex — or any custom agent setup.

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

## Quick Start — `optimize()`

The simplest way to use AgentOpt. No `ModelProxy`, no `invoke_fn`, no async wrappers. Just pass your agent and the models you want to test:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from agentopt import optimize

# 1. Build your agent normally
agent = FunctionAgent(
    tools=[multiply, add],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a math helper.",
)

# 2. One call to optimize — evaluates models in parallel
results = optimize(
    agent=agent,
    models=["gpt-4o-mini", "gpt-4o"],
    eval_fn=lambda expected, actual: expected.lower() in str(actual).lower(),
    dataset=[({"user_msg": "What is 2+2?"}, "4"), ...],
)

# 3. Agent LLM is automatically set to the best model
print(results.get_best())
print(agent.llm.model)  # -> best model name
```

`optimize()` works with any framework that has an `.llm` attribute and a `.kickoff()`, `.invoke()`, or `.run()` method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Any` | Agent with `.llm` attribute and `.kickoff()`/`.invoke()`/`.run()` |
| `models` | `list[str]` | Model name strings to evaluate |
| `eval_fn` | `(str, Any) -> bool \| float` | Compares expected answer to actual output |
| `dataset` | `list[tuple[Any, str]]` | `(input_data, expected_answer)` tuples |
| `parallel` | `bool` | Evaluate models concurrently (default: `True`) |
| `max_workers` | `int \| None` | Max threads for parallel mode (default: `len(models)`) |

## Advanced — `ModelProxy` + `ModelSelector`

For more control (multi-LLM optimization, custom invoke functions), use the `ModelProxy` workflow:

1. **Wrap** your LLM with `ModelProxy`
2. **Build** your agent as usual (the proxy is transparent)
3. **Run** `ModelSelector.select_best()` — with optional `parallel=True`

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

# 4. Run optimization
selector = ModelSelector(
    models={llm: ["openai/gpt-4o-mini", "openai/gpt-4o"]},
    eval_fn=lambda expected, actual: expected.lower() in str(actual).lower(),
    dataset=dataset,
    agent=crew,  # auto-detects crew.kickoff()
)

# Sequential evaluation (swaps proxy in-place)
results = selector.select_best()

# Or parallel evaluation (clones agent per combination, uses thread pool)
results = selector.select_best(parallel=True, max_workers=4)

print(results.get_best())
```

The CrewAI example can be run from the command line:

```bash
# Sequential
uv run python examples/crewai_example.py single

# Parallel
uv run python examples/crewai_example.py single --parallel

# Multi-agent examples
uv run python examples/crewai_example.py multi --parallel
uv run python examples/crewai_example.py multi-llm --parallel
```

### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLM
llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# 2. Build your agent normally
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# 3. Run optimization
selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=executor,  # auto-detects executor.invoke()
)
results = selector.select_best(parallel=True)
```

### LlamaIndex

**Recommended:** Use `optimize()` (see Quick Start above). It handles LlamaIndex's async workflow and Pydantic validation automatically.

If you need `ModelProxy` for multi-LLM scenarios:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from agentopt import ModelProxy, ModelSelector

initial_llm = OpenAI(model="gpt-4o-mini")
llm_proxy = ModelProxy(initial_llm)

agent = FunctionAgent(
    tools=[multiply],
    llm=initial_llm,  # actual LLM, not proxy (Pydantic validation)
    system_prompt="You are a helpful assistant."
)

# Custom invoke function swaps the LLM before each run
async def invoke_with_proxy(input_data):
    agent.llm = llm_proxy.get_model()
    return await agent.run(**input_data)

selector = ModelSelector(
    models={llm_proxy: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=lambda expected, actual: expected in str(actual),
    dataset=[({"user_msg": "What is 2 * 3?"}, "6"), ...],
    invoke_fn=invoke_with_proxy,
)
results = selector.select_best()
```

**Note:** LlamaIndex agents use Pydantic validation which prevents passing `ModelProxy` directly. The `invoke_fn` pattern works around this. Parallel mode (`select_best(parallel=True)`) requires `agent=` instead of `invoke_fn=`.

### Custom Agent / Any Framework

For agents that don't use `.kickoff()` or `.invoke()`, pass a custom `invoke_fn`:

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

### ModelProxy

A transparent wrapper around any LLM object that enables model swapping without rebuilding agents. It forwards all attribute access and method calls to the underlying model.

```python
from agentopt import ModelProxy

proxy = ModelProxy(original_llm)

# Behaves exactly like original_llm
proxy.invoke(...)   # forwarded
proxy.temperature   # forwarded

# Swap the underlying model (by string or full object)
proxy.set_model("gpt-4o")           # updates the model name in-place
proxy.set_model(new_llm_instance)   # replaces entirely

# Inspect
proxy.get_model()  # returns the current underlying model
```

**Why is this needed?** Agent frameworks capture the model reference at construction time. Without `ModelProxy`, you'd need to rebuild the entire agent pipeline for every model you want to test. The proxy provides a stable reference that agents hold, while the actual model behind it can be swapped.

```
Agent --> ModelProxy --> LLM (gpt-4o-mini)
                    --> LLM (gpt-4o)        # swapped, agent unchanged
```

**String-based swapping:** When you pass a string to `set_model()`, the proxy updates the model name field on the existing model object (e.g., `.model` for CrewAI, `.model_name` for LangChain). For Pydantic-based models, it uses `model_copy()` to create an immutable update.

### ModelSelector

Evaluates your agent across model combinations and selects the best one based on accuracy and latency. Supports both sequential and parallel evaluation.

```python
from agentopt import ModelSelector

selector = ModelSelector(
    models=models,       # {ModelProxy: [candidate_models]}
    eval_fn=eval_fn,     # (expected, actual) -> bool | float
    dataset=dataset,     # [(input_data, expected_answer), ...]
    agent=agent,         # or invoke_fn=callable
)

# Sequential: swaps model proxy in-place, evaluates one combination at a time
results = selector.select_best()

# Parallel: clones agent per combination, evaluates concurrently via thread pool
results = selector.select_best(parallel=True, max_workers=4)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | `dict[ModelProxy, list]` | Maps each proxy to its candidate models (strings or objects) |
| `eval_fn` | `(str, Any) -> bool \| float` | Compares expected answer to actual output. Returns bool or float score (higher is better) |
| `dataset` | `list[tuple[Any, str]]` | List of `(input_data, expected_answer)` tuples. `input_data` is passed directly to the agent's invoke method |
| `agent` | `Any` | Agent object with `.kickoff()` (CrewAI), `.invoke()` (LangChain), or `.run()` (LlamaIndex). Mutually exclusive with `invoke_fn` |
| `invoke_fn` | `callable` | Custom function `(input_data) -> result`. Mutually exclusive with `agent`. Not compatible with `parallel=True` |

**`select_best()` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `parallel` | `bool` | Evaluate combinations concurrently (default: `False`) |
| `max_workers` | `int \| None` | Max threads for parallel mode (default: number of combinations) |

### How parallel evaluation works

When `parallel=True`, `select_best()`:
1. Detects which agent attribute each `ModelProxy` maps to
2. Creates independent agent copies via Pydantic `model_copy(deep=False)`, each with a fresh LLM variant
3. Evaluates all copies concurrently using `ThreadPoolExecutor`
4. Sets the original proxies to the winning combination

Each thread gets its own agent instance with its own LLM — no shared state, no conflicts. If cloning fails, it falls back to sequential evaluation automatically.

**Note:** Parallel mode requires `agent=` (not `invoke_fn=`), since the agent must be cloneable.

### Multi-Agent / Multi-LLM Optimization

When your pipeline uses multiple LLMs (e.g., different agents with different models), create a separate `ModelProxy` for each and pass them all to `ModelSelector`. It evaluates the **Cartesian product** of all candidate combinations.

```python
researcher_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
coder_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# ... build agents with these proxies ...

selector = ModelSelector(
    models={
        researcher_llm: ["gpt-4o-mini", "gpt-4o"],
        coder_llm: ["gpt-4o-mini", "gpt-4o"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
    invoke_fn=my_pipeline,
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
│   ├── optimize.py              # optimize() and parallel_select() — thin wrappers
│   ├── model_proxy.py           # ModelProxy for transparent model swapping
│   ├── model_factory.py         # create_model_from_string (LangChain model creation)
│   ├── base_models.py           # Type aliases (EvalFn, ModelSpec, ModelsConfig)
│   └── model_selection/
│       ├── __init__.py          # Exports ModelSelector
│       ├── base.py              # BaseModelSelector, parallel utilities, result types
│       └── brute_force.py       # BruteForceModelSelector (sequential + parallel)
├── examples/
│   ├── crewai_example.py              # CrewAI: single-agent, multi-agent (CLI)
│   ├── langchain_example.py           # LangChain: single-agent, multi-agent
│   ├── llamaindex_example.py          # LlamaIndex: ModelProxy + invoke_fn approach
│   ├── llamaindex_optimize_example.py # LlamaIndex: optimize() — simplest API
│   └── datasets/
│       └── math_problems.jsonl
└── pyproject.toml
```

## Environment Setup

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY=your_key_here
```

The `model_factory` module (used for LangChain string-based model creation) also supports OpenRouter as a fallback:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Or use a `.env` file:
```bash
cp .env.example .env
```

## API Reference

### Exports

```python
from agentopt import (
    # Simple API (recommended)
    optimize,                # Find best model — parallel, no ModelProxy needed
    parallel_select,         # Parallel evaluation with ModelProxy (convenience wrapper)

    # Core
    ModelProxy,              # Transparent LLM proxy
    ModelSelector,           # Brute-force model selector (sequential + parallel)
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