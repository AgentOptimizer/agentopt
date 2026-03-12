# AgentOpt - Optimization for AI Agents

A framework-agnostic toolkit for optimizing LLM-powered agents. Evaluate and select the best models for your agents across frameworks like CrewAI, LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, and AG2 — or any custom agent setup.

> **Note: This project is in early development. APIs are subject to change.**

## Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync base dependencies
uv sync

# Framework-specific extras
uv sync --extra crewai        # CrewAI
uv sync --extra llamaindex    # LlamaIndex
uv sync --extra ag2           # AG2 (AutoGen 2)
uv sync --extra openai-agents # OpenAI Agents SDK
uv sync --extra claude-agent-sdk  # Claude Agent SDK
```

## Quick Start

AgentOpt works in three steps:

1. **Wrap** your LLM with `ModelProxy`
2. **Build** your agent as usual (the proxy is transparent)
3. **Run** `ModelSelector` (default brute-force) to find the best model

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
    models={llm: ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=executor,
)
results = selector.select_best()
```

### LangGraph

LangGraph graphs are compiled state machines — use `invoke_fn` instead of `agent=`:

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

# 3. Run optimization — parallel mode uses thread-local model overrides automatically
selector = ModelSelector(
    models={proxy: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    invoke_fn=invoke_fn,
)
results = selector.select_best(parallel=True)
```

### LlamaIndex

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from agentopt import ModelProxy, ModelSelector
from agentopt.model_proxy.framework_specific_implementation.llamaindex import build_llamaindex_llm

# 1. Create a real LLM (ModelProxy can't be passed directly due to Pydantic v2)
real_llm = build_llamaindex_llm("gpt-4o-mini")

# 2. Build your agent normally
agent = FunctionAgent(
    tools=my_tools,
    llm=real_llm,
    system_prompt="You are a helpful assistant.",
)
workflow = AgentWorkflow(agents=[agent], root_agent=agent.name)

# 3. Wrap the LLM after agent creation
llm = ModelProxy(real_llm)

selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=workflow,  # auto-detects workflow.run() (async)
)
results = selector.select_best()
```

### OpenAI Agents SDK

```python
from types import SimpleNamespace
from agents import Agent
from agentopt import ModelProxy, ModelSelector

# 1. Wrap a model — ModelProxy is registered as a Model ABC subclass
proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))

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

### Claude Agent SDK

The Claude SDK is functional (no persistent agent object), so use `invoke_fn`:

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, query
from agentopt import ModelProxy, ModelSelector

proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))

async def _query_async(prompt, options):
    result = ""
    async for msg in query(prompt=prompt, options=options):
        if hasattr(msg, "result"):
            result = msg.result
    return result

def invoke_fn(input_data):
    return asyncio.run(_query_async(input_data["input"], proxy))

selector = ModelSelector(
    models={proxy: ["haiku", "sonnet"]},  # Claude SDK uses short aliases
    eval_fn=my_eval_fn,
    dataset=dataset,
    invoke_fn=invoke_fn,
)
results = selector.select_best(parallel=True)
```

### AG2 (AutoGen 2)

AG2 validates `llm_config` and rejects `ModelProxy`, so AgentOpt patches the validation at import time. Just pass an `LLMConfig` to `ModelProxy`:

```python
from autogen import ConversableAgent, LLMConfig
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLMConfig — AgentOpt converts it to a mutable wrapper internally
proxy = ModelProxy(LLMConfig({"model": "gpt-4o-mini", "api_key": "..."}))

# 2. Build agent with the proxy as llm_config (patched validation accepts it)
agent = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=proxy,
    human_input_mode="NEVER",
)

# 3. Run optimization — auto-detects agent.run()
selector = ModelSelector(
    models={proxy: ["gpt-4o-mini", "anthropic/claude-sonnet-4-20250514"]},
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

**The problem it solves:** Without adapters, every time we added a new framework we had to edit three separate files — the proxy, the selector's `__init__`, and the parallel cloning logic. All three had the same `if is_crewai: ... elif is_langchain: ...` chains.

**The solution:** One `FrameworkAdapter` class per framework. Each adapter answers four questions:

```
1. detect()              — Is this agent object mine?
2. get_invoke_fn()       — How do I call the agent to get a result?
3. register_with_proxy() — How do I wire sync callbacks so set_model() propagates?
4. clone_for_parallel()  — How do I make a deep-safe independent copy for a thread?
```

**How it fits together at runtime:**

```
User calls: ModelSelector(agent=my_crew, ...)
                |
                v
         get_adapter(my_crew)
                |  iterates _REGISTRY, calls each adapter.detect()
                |  CrewAIAdapter.detect() -> type(crew).__module__.startswith("crewai") -> True
                v
         CrewAIAdapter
          +-- get_invoke_fn()       -> returns crew.kickoff (bound method)
          +-- register_with_proxy() -> adds sync closures to proxy._sync_callbacks
          +-- clone_for_parallel()  -> model_copy(deep=False) + clone_crew_agents()
```

**Self-registration:** Each framework file registers its own adapter at import time:

```python
# bottom of crewai.py
register_adapter(CrewAIAdapter())

# bottom of langchain_compat.py
register_adapter(LangChainAdapter())
# ... etc.
```

**Adding a new framework** is just one new file in `model_proxy/framework_specific_implementation/`:

```python
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
| `agent` | `Any` | Agent object (CrewAI Crew, LangChain AgentExecutor, LlamaIndex AgentWorkflow, OpenAI SDK Agent, AG2 ConversableAgent). Mutually exclusive with `invoke_fn` |
| `invoke_fn` | `callable` | Custom function `(input_data) -> result`. Use for LangGraph, Claude SDK, or custom pipelines. Mutually exclusive with `agent` |

**`select_best()` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `parallel` | `bool` | Evaluate combinations concurrently (default: `False`) |
| `max_workers` | `int \| None` | Max threads for parallel mode (default: number of combinations) |

### Selection Strategies

AgentOpt includes several model selection strategies:

- **`BruteForceModelSelector`** (default, aliased as `ModelSelector`) — Grid search over the full Cartesian product of all candidate model combinations. Thorough but scales as O(n^k) where n is models per proxy and k is the number of proxies.

- **`RandomSearchModelSelector`** — Random subset search over the Cartesian product: samples a fraction of all combinations and evaluates only that subset. Useful when brute force is too expensive but you still want broad coverage. Supports the same sequential and parallel modes as brute force via `select_best(parallel=...)`.

- **`HillClimbingModelSelector`** — Local-search / hill-climbing strategy using model quality/speed rankings. Starts from an initial combination and iteratively swaps one model at a time, keeping improvements. Much faster for large search spaces.

- **`ArmEliminationModelSelector`** — Bandit-style successive elimination strategy that evaluates combinations in rounds with growing batch sizes and drops statistically dominated arms using confidence bounds. Often reduces total API calls versus brute force.

- **`HyperbandModelSelector`** — Bandit-style full Hyperband algorithm over the dataset, treating the number of samples as resource and running multiple successive-halving brackets with different starting budgets. The key hyperparameter is the reduction factor `η` (`reduction_factor`).

- **`BayesianOptimizationModelSelector`** — Gaussian process-based optimization that models the accuracy surface over model combinations and uses an acquisition function to select the most promising combination to evaluate next. Efficient for large search spaces.

```python
from agentopt import (
    BaseModelSelector,
    BruteForceModelSelector,
    RandomSearchModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
    HyperbandModelSelector,
)

# Brute force (tests all combinations)
selector = BruteForceModelSelector(models=models, eval_fn=eval_fn, dataset=dataset, agent=agent)

# Random search (tests a sampled subset)
selector = RandomSearchModelSelector(
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
    sample_fraction=0.25,
)

# Hill climbing (smart search)
selector = HillClimbingModelSelector(models=models, eval_fn=eval_fn, dataset=dataset, agent=agent)

# Arm elimination (bandit-style successive elimination)
selector = ArmEliminationModelSelector(models=models, eval_fn=eval_fn, dataset=dataset, agent=agent)

# Hyperband (multi-bracket successive halving over dataset samples)
selector = HyperbandModelSelector(
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
    reduction_factor=3.0,
)
```

`RandomSearchModelSelector` samples without replacement from the full search space. Set `sample_fraction` to a value in `(0, 1]`; for example, `0.25` evaluates 25% of all combinations.

Example CLI usage:

```bash
python examples/ag2_example.py single --selector random_search --sample-fraction 0.25

# Hyperband with custom reduction factor
python examples/ag2_example.py single --selector hyperband --reduction-factor 3.0
```

### How Parallel Evaluation Works

When `parallel=True`, `select_best()`:

**Current selector support**

- **Supports `parallel=True` (actually parallel):**
  - `BruteForceModelSelector`
  - `RandomSearchModelSelector`
  - `ArmEliminationModelSelector`
  - `HyperbandModelSelector`

> **Experimental note:** `HillClimbingModelSelector`, `ArmEliminationModelSelector`, `HyperbandModelSelector`, and `BayesianOptimizationModelSelector` are currently experimental and their parallel behavior and APIs may change in future releases.

**With `agent=`:**
1. Detects the framework via the adapter registry
2. Creates independent agent copies via `adapter.clone_for_parallel()`, each with a fresh LLM
3. Evaluates all copies concurrently using `ThreadPoolExecutor`
4. Sets the original proxies to the winning combination

**With `invoke_fn=`:**
1. Each thread sets per-thread model overrides on the `ModelProxy` objects captured in the closure
2. The proxy's `_get_effective_model()` returns the thread-local model, so each thread sees its own model
3. Evaluates all combinations concurrently using `ThreadPoolExecutor`

Each thread gets its own model instances via thread-local storage — no shared state, no conflicts.

### Multi-Agent / Multi-LLM Optimization

When your pipeline uses multiple LLMs (e.g., different agents with different models), create a separate `ModelProxy` for each and pass them all to `ModelSelector` (or another selector). It evaluates the **Cartesian product** of all candidate combinations.

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
results = selector.select_best(parallel=True)
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
| `input_tokens` | `int` | Total input tokens consumed |
| `output_tokens` | `int` | Total output tokens consumed |
| `attribute` | `str` | Grouping label (e.g., `"combination"`) |
| `is_best` | `bool` | Whether this was the best result |

## Framework Support Matrix

| Framework | Proxy works directly? | Invoke method | Cross-provider? | Multi-agent | Parallel |
|-----------|----------------------|---------------|-----------------|-------------|----------|
| CrewAI | Yes (duck typing) | `.kickoff()` | Yes | Yes | Yes (adapter) |
| LangChain | Yes (duck typing) | `.invoke()` | Yes | Yes (multi-llm) | Yes (adapter) |
| LangGraph | N/A (uses `invoke_fn`) | graph `.invoke()` | Yes | Yes | Yes (thread-local) |
| LlamaIndex | No (Pydantic strict) | `.run()` (async) | Yes | Yes | Yes (adapter) |
| OpenAI SDK | Yes (ABC virtual subclass) | `Runner.run_sync()` | OpenAI only | Yes (multi-llm) | Yes (adapter / thread-local) |
| Claude SDK | N/A (uses `invoke_fn`) | `query()` (async) | Claude only | Yes | Yes (thread-local) |
| AG2 | No (patched validation) | `.run()` | OpenAI + Anthropic | Yes | Yes (adapter / thread-local) |

**Known limitations:**
- **OpenAI SDK** only supports OpenAI models natively
- **Claude SDK** only supports Claude models (uses short aliases: `"haiku"`, `"sonnet"`, `"opus"`)
- **AG2** supports OpenAI and Anthropic models natively; other providers not yet supported
- **LangGraph** does not have a single-agent example (use LangChain for single-agent)
- **OpenRouter** has compatibility issues with most frameworks; use native OpenAI and Anthropic API keys instead

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
│   ├── model_proxy/
│   │   ├── proxy.py             # ModelProxy — transparent proxy, set_model(), register()
│   │   ├── adapter.py           # FrameworkAdapter ABC + registry (get_adapter, register_adapter)
│   │   ├── constants.py         # Framework detection helpers + MODEL_FIELDS
│   │   ├── token_tracking.py    # TokenAccumulator + extract_usage()
│   │   ├── builders.py          # Generic LLM rebuild helpers
│   │   └── framework_specific_implementation/
│   │       ├── crewai.py        # CrewAI support + CrewAIAdapter
│   │       ├── langchain_compat.py  # LangChain support + LangChainAdapter
│   │       ├── llamaindex.py    # LlamaIndex support + LlamaIndexAdapter + build_llamaindex_llm
│   │       ├── openai_sdk.py    # OpenAI Agents SDK support + OpenAISDKAdapter
│   │       └── ag2.py           # AG2 support + AG2Adapter + ProxyAwareWrapper
│   └── model_selection/
│       ├── base.py              # BaseModelSelector, ModelResult, SelectionResults
│       ├── brute_force.py       # BruteForceModelSelector (default ModelSelector)
│       ├── random_search.py     # RandomSearchModelSelector (sampled brute-force)
│       ├── hill_climbing.py     # HillClimbingModelSelector (experimental)
│       ├── arm_elimination.py   # ArmEliminationModelSelector (experimental)
│       ├── hyperband.py         # HyperbandModelSelector (experimental)
│       ├── bayesian_optimization.py  # BayesianOptimizationModelSelector (experimental)
│       └── utils.py             # Compat re-export of extract_prompt
├── examples/
│   ├── crewai_example.py        # CrewAI: single, multi-agent, multi-LLM
│   ├── langchain_example.py     # LangChain: single, multi-LLM
│   ├── langgraph_example.py     # LangGraph: multi-agent, multi-LLM
│   ├── llamaindex_example.py    # LlamaIndex: single, multi-agent, multi-LLM
│   ├── openai_sdk_example.py    # OpenAI SDK: single, multi-LLM
│   ├── claude_sdk_example.py    # Claude SDK: single, multi-agent, multi-LLM
│   ├── ag2_example.py           # AG2: single, multi-agent, multi-LLM
│   └── datasets/
│       └── math_problems.jsonl
└── pyproject.toml
```

## Running Examples

All examples use a consistent CLI:

```bash
# CrewAI
uv run python examples/crewai_example.py single
uv run python examples/crewai_example.py multi-llm --parallel

# LangChain
uv run python examples/langchain_example.py
uv run python examples/langchain_example.py --parallel

# LangGraph
uv run python examples/langgraph_example.py multi
uv run python examples/langgraph_example.py multi-llm --parallel

# LlamaIndex
uv run --extra llamaindex python examples/llamaindex_example.py single
uv run --extra llamaindex python examples/llamaindex_example.py multi-llm --parallel

# OpenAI Agents SDK
uv run --extra openai-agents python examples/openai_sdk_example.py single
uv run --extra openai-agents python examples/openai_sdk_example.py multi-llm --parallel

# Claude Agent SDK
uv run --extra claude-agent-sdk python examples/claude_sdk_example.py single
uv run --extra claude-agent-sdk python examples/claude_sdk_example.py multi-llm --parallel

# AG2
uv run --extra ag2 python examples/ag2_example.py single
uv run --extra ag2 python examples/ag2_example.py multi-llm --parallel
```

Common flags: `--parallel`, `--dataset <filename>`, `--no-plot`

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

## API Reference

### Exports

```python
from agentopt import (
    # Core
    ModelProxy,              # Transparent LLM proxy with auto-framework-detection
    ModelSelector,           # Brute-force model selector (default)
    BruteForceModelSelector, # Explicit brute-force selector (grid search over all combinations)
    RandomSearchModelSelector, # Random subset search over model combinations
    HillClimbingModelSelector, # Hill-climbing selector (local search over combinations)
    ArmEliminationModelSelector, # Arm-elimination selector (bandit-style successive elimination)
    HyperbandModelSelector,  # Hyperband selector (bandit-style, multi-bracket successive halving over dataset samples)
    BayesianOptimizationModelSelector, # Bayesian optimization selector (Bayesian search over combinations)
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
