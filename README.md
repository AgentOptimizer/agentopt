# AgentOpt - Optimization for AI Agents

A framework-agnostic toolkit for optimizing LLM-powered agents. Evaluate and select the best models for your agents across frameworks like CrewAI, LangChain, LangGraph, LlamaIndex, and AG2 — or any custom agent setup.

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

# For AG2 (AutoGen 2) support
uv sync --extra ag2
```

## Environment Setup

Set API keys for the providers you want to use in a `.env` file:

```bash
cp .env.example .env
```

Or export them directly:

```bash
# Direct API access (set whichever you need)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...

# AWS Bedrock
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=wJalr...
export AWS_DEFAULT_REGION=us-east-1

# Proxy fallbacks (optional)
export LITELLM_API_KEY=sk-...
export LITELLM_API_BASE=http://localhost:4000
export OPENROUTER_API_KEY=sk-or-...
```

**Note:** If a variable exists in both your shell environment and `.env` file, the shell environment takes priority. The `.env` file only fills in variables that aren't already set.

## Multi-Provider Support

AgentOpt supports any LLM provider out of the box. You can mix and match models from different providers in a single optimization run — the framework handles cross-provider swapping automatically by rebuilding the LLM object for each provider.

### Supported Providers

| Provider | Model name format | How it connects |
|----------|------------------|-----------------|
| **OpenAI** | `gpt-4o-mini`, `openai/gpt-4o` | Direct API via `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-haiku-20240307`, `anthropic/claude-3-5-sonnet-20241022` | Direct API via `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini-1.5-flash`, `google/gemini-1.5-pro` | Direct API via `GOOGLE_API_KEY` |
| **AWS Bedrock** | `bedrock/meta.llama3-3-70b-instruct-v1:0`, `bedrock/mistral.mistral-large-2407-v1:0` | AWS SDK via `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_DEFAULT_REGION` |
| **LiteLLM** | `litellm/gpt-4o` | Proxy server via `LITELLM_API_KEY` + `LITELLM_API_BASE` |
| **OpenRouter** | (automatic fallback) | Routes any model via `OPENROUTER_API_KEY` |

### How Provider Routing Works

Just use the model name — AgentOpt detects the provider automatically:

```python
selector = ModelSelector(
    models={
        llm: [
            "gpt-4o-mini",                                    # → OpenAI
            "claude-3-haiku-20240307",                        # → Anthropic
            "gemini-1.5-flash",                               # → Google
            "bedrock/meta.llama3-3-70b-instruct-v1:0",       # → AWS Bedrock
        ]
    },
    ...
)
```

Provider detection uses prefixes (`bedrock/`, `litellm/`, `openai/`, `google/`, `anthropic/`) and keyword matching (`claude`, `gemini`, `gpt`). If a native API key isn't available, it falls back through **LiteLLM → OpenRouter** automatically.

## Quick Start — `ModelProxy` + `ModelSelector`

Use the `ModelProxy` workflow:

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

# 4. Run optimization — mix providers freely
selector = ModelSelector(
    models={llm: ["openai/gpt-4o-mini", "anthropic/claude-3-haiku-20240307"]},
    eval_fn=lambda expected, actual: expected.lower() in str(actual).lower(),
    dataset=dataset,
    agent=crew,  # auto-detects crew.kickoff()
)

# Sequential evaluation
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

# 3. Run optimization — cross-provider works automatically
selector = ModelSelector(
    models={llm: [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-haiku-20240307",
        "bedrock/meta.llama3-3-70b-instruct-v1:0",
    ]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=executor,  # auto-detects executor.invoke()
)
results = selector.select_best(parallel=True)
```

### LlamaIndex

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

### AG2 (AutoGen 2)

AG2 validates `llm_config` and rejects `ModelProxy`, so use a mutable config wrapper. The agent receives a real `LLMConfig` backed by mutable `config_list`; `ModelProxy` wraps the wrapper and updates the model in place when `set_model()` is called:

```python
from autogen import ConversableAgent, LLMConfig
from agentopt import ModelSelector, ModelProxy

# 1. Mutable wrapper (AG2 rejects ModelProxy as llm_config)
class AG2ConfigWrapper:
    def __init__(self, model="gpt-4o-mini"):
        self.config_list = [{"api_type": "openai", "model": model, "api_key": "..."}]
    @property
    def model(self):
        return self.config_list[0]["model"]
    @model.setter
    def model(self, value):
        self.config_list[0]["model"] = value

config_wrapper = AG2ConfigWrapper("gpt-4o-mini")
llm_config = LLMConfig(config_list=config_wrapper.config_list)
llm_config_proxy = ModelProxy(config_wrapper)

# 2. Build agent with real LLMConfig
agent = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 3. Custom invoke_fn (AG2 uses run(), not invoke/kickoff)
def invoke_fn(input_data):
    response = agent.run(message=input_data["input"], max_turns=2, user_input=False)
    for _ in response.events:
        pass
    return response.summary or (response.messages[-1].content if response.messages else "")

# 4. Run optimization (proxy updates config_list[0]["model"] in place)
selector = ModelSelector(
    models={llm_config_proxy: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    invoke_fn=invoke_fn,
)
results = selector.select_best()
```

See `examples/ag2_example.py` for a complete example. Run it with:

```bash
uv run python examples/ag2_example.py single   # single-agent
uv run python examples/ag2_example.py multi    # multi-agent (researcher + coder)
```

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
proxy.set_model("claude-3-haiku-20240307")  # cross-provider: rebuilds the LLM automatically

# Inspect
proxy.get_model()  # returns the current underlying model
```

**Why is this needed?** Agent frameworks capture the model reference at construction time. Without `ModelProxy`, you'd need to rebuild the entire agent pipeline for every model you want to test. The proxy provides a stable reference that agents hold, while the actual model behind it can be swapped.

```
Agent --> ModelProxy --> LLM (gpt-4o-mini)
                    --> LLM (claude-3-haiku)   # swapped, agent unchanged
```

**Cross-provider swapping:** When you swap to a model from a different provider (e.g., OpenAI → Anthropic), the proxy fully rebuilds the LLM object using the correct provider class, API key, and defaults. No settings leak between providers.

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
| `agent` | `Any` | Agent object with `.kickoff()` (CrewAI) or `.invoke()` (LangChain). For AG2, use `invoke_fn` instead. Mutually exclusive with `invoke_fn` |
| `invoke_fn` | `callable` | Custom function `(input_data) -> result`. Mutually exclusive with `agent` |
| `agent` | `Any` | Agent object with `.kickoff()` (CrewAI), `.invoke()` (LangChain), or `.run()` (LlamaIndex). Mutually exclusive with `invoke_fn` |
| `invoke_fn` | `callable` | Custom function `(input_data) -> result`. Mutually exclusive with `agent`. Not compatible with `parallel=True` |

**`select_best()` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `parallel` | `bool` | Evaluate combinations concurrently (default: `False`) |
| `max_workers` | `int \| None` | Max threads for parallel mode (default: number of combinations) |

### Selection Strategies

AgentOpt includes two model selection strategies:

- **`BruteForceModelSelector`** (default, aliased as `ModelSelector`) — Evaluates the Cartesian product of all candidate model combinations. Thorough but scales as O(n^k) where n is models per proxy and k is the number of proxies.

- **`HillClimbingModelSelector`** — Neighbor-based search using model quality/speed rankings. Starts from an initial combination and iteratively swaps one model at a time, keeping improvements. Much faster for large search spaces.

```python
from agentopt import BruteForceModelSelector, HillClimbingModelSelector

# Brute force (tests all combinations)
selector = BruteForceModelSelector(models=models, eval_fn=eval_fn, dataset=dataset, agent=agent)

# Hill climbing (smart search)
selector = HillClimbingModelSelector(models=models, eval_fn=eval_fn, dataset=dataset, agent=agent)
```

### How Parallel Evaluation Works

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
│   ├── model_proxy.py           # ModelProxy for transparent model swapping
│   ├── model_factory.py         # Multi-provider LLM factory with fallback chain
│   ├── model_topology.py        # Model quality/speed rankings for hill climbing
│   ├── base_models.py           # Type aliases (EvalFn, ModelSpec, ModelsConfig)
│   └── model_selection/
│       ├── __init__.py          # Exports ModelSelector
│       ├── base.py              # BaseModelSelector, parallel utilities, result types
│       ├── brute_force.py       # BruteForceModelSelector (sequential + parallel)
│       └── hill_climbing.py     # HillClimbingModelSelector (neighbor-based search)
├── examples/
│   ├── crewai_example.py        # CrewAI: single-agent, multi-agent, hierarchical
│   ├── langchain_example.py     # LangChain: single-agent, multi-agent with chaining
│   ├── ag2_example.py           # AG2: single-agent, multi-agent with invoke_fn
│   ├── crewai_example.py              # CrewAI: single-agent, multi-agent (CLI)
│   ├── langchain_example.py           # LangChain: single-agent, multi-agent
│   ├── llamaindex_example.py          # LlamaIndex: ModelProxy + invoke_fn approach
│   └── datasets/
│       └── math_problems.jsonl
└── pyproject.toml
```

## API Reference

### Exports

```python
from agentopt import (
    # Core
    ModelProxy,              # Transparent LLM proxy
    ModelSelector,           # Default model selector (BruteForceModelSelector)
    BruteForceModelSelector, # Brute-force model selector (sequential + parallel)
    HillClimbingModelSelector, # Hill-climbing model selector
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
