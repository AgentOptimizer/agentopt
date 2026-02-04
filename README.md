# AgentOpt - Model Selection for AI Agents

A framework-agnostic package for evaluating and selecting optimal LLM models for AI agents. Supports CrewAI, LangChain, and extensible to other frameworks.

> **⚠️ Note: This project is in early development. Several components are subject to change—see [Current Limitations](#current-limitations) below.**

## Installation

Using `uv` (recommended):

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run LangChain example (works with default dependencies)
uv run python examples/langchain_example.py

# For CrewAI examples, install optional dependencies
uv sync --extra crewai
uv run python examples/crewai_example.py
```

## Project Structure

```
agentopt/
├── src/agentopt/
│   ├── __init__.py              # Public API exports
│   ├── types.py                 # Core type definitions (Message, EvaluationTask)
│   ├── model_proxy.py           # ModelProxy for dynamic model swapping
│   ├── model_factory.py         # Model creation utilities
│   ├── load_dataset.py          # Dataset loading (JSONL → EvaluationTask)
│   ├── invoker/                 # Framework adapters
│   │   ├── base.py              # InvokerProtocol definition
│   │   ├── langchain.py         # LangchainInvoker, ChainedLangchainInvoker
│   │   └── crewai.py            # CrewInvoker
│   └── model_selection/         # Model selection logic
│       ├── base.py              # BaseModelSelector, ModelResult, SelectionResults
│       └── model_selection.py   # ModelSelector implementation
├── examples/
│   ├── crewai_example.py        # CrewAI usage examples
│   ├── langchain_example.py     # LangChain usage examples
│   └── datasets/
│       └── math_problems.jsonl
└── pyproject.toml
```

## Environment Setup

Set your API keys as environment variables:

```bash
export OPENROUTER_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here        # Optional, for OpenAI direct
export ANTHROPIC_API_KEY=your_key_here     # Optional, for Anthropic direct
export GOOGLE_API_KEY=your_key_here        # Optional, for Google direct
```

Or use a `.env` file:
```bash
cp .env.example .env
# Edit .env with your keys
```

## Core Concepts

### 1. ModelProxy

A transparent wrapper that enables dynamic model swapping without recreating agents:

```python
from agentopt import ModelProxy

# Wrap your LLM with ModelProxy
proxy = ModelProxy(initial_model)

# Use proxy in your agent (it behaves exactly like the original model)
agent = create_agent(model=proxy, ...)

# Later, swap the underlying model
proxy.set_model(new_model)  # Agent now uses new_model
```

### 2. Invokers

Framework-specific wrappers that provide a uniform interface for agent invocation:

```python
from agentopt import CrewInvoker, LangchainInvoker

# For CrewAI
invoker = CrewInvoker(crew)

# For LangChain
invoker = LangchainInvoker(agent_executor)

# Uniform interface
result = invoker.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

### 3. ModelSelector

Evaluates multiple models on a dataset and selects the best performer:

```python
from agentopt import ModelSelector

selector = ModelSelector(
    invoker=invoker,
    models={proxy: ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]},
    accuracy_fn=lambda expected, actual: expected.lower() in actual.lower(),
    dataset_dir="path/to/dataset",
)

results = selector.select_best()
print(results.get_best())  # Best model across all
results.to_csv("results.csv")
```

## Usage Examples

### CrewAI Example

```python
from crewai import Agent, Task, Crew, LLM
from agentopt import ModelProxy, CrewInvoker, ModelSelector

# 1. Create LLM wrapped with ModelProxy
llm = LLM(model="openai/gpt-4o", base_url="https://openrouter.ai/api/v1")
proxy = ModelProxy(llm)

# 2. Create agent using the proxy
agent = Agent(
    role="Researcher",
    goal="Research and answer questions",
    backstory="You are a helpful research assistant.",
    llm=proxy,
)

# 3. Create task and crew
task = Task(
    description="Answer: {input}",
    expected_output="A clear answer",
    agent=agent,
)
crew = Crew(agents=[agent], tasks=[task])

# 4. Wrap with invoker and run model selection
invoker = CrewInvoker(crew)
selector = ModelSelector(
    invoker=invoker,
    models={proxy: ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash"]},
    accuracy_fn=lambda expected, actual: expected.lower() in actual.lower(),
    dataset_dir="examples/datasets",
)

results = selector.select_best()
```

### LangChain Example

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from agentopt import ModelProxy, LangchainInvoker, ModelSelector

# 1. Create model wrapped with ModelProxy
model = ChatOpenAI(model="gpt-4o")
proxy = ModelProxy(model)

# 2. Create agent using the proxy
agent = create_react_agent(llm=proxy, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# 3. Wrap with invoker and run model selection
invoker = LangchainInvoker(executor)
selector = ModelSelector(
    invoker=invoker,
    models={proxy: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]},
    accuracy_fn=my_accuracy_fn,
    dataset_dir="datasets",
)

results = selector.select_best()
```

### Multi-Agent / Multi-LLM Example

```python
# Different proxies for different agents
researcher_proxy = ModelProxy(LLM(model="openai/gpt-4o"))
coder_proxy = ModelProxy(LLM(model="anthropic/claude-3.5-sonnet"))

researcher = Agent(role="Researcher", llm=researcher_proxy, ...)
coder = Agent(role="Coder", llm=coder_proxy, ...)

# Optimize each LLM independently
selector = ModelSelector(
    invoker=invoker,
    models={
        researcher_proxy: ["openai/gpt-4o", "google/gemini-2.0-flash"],
        coder_proxy: ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
    },
    ...
)
```

## Dataset Format

JSONL files with question/output pairs:

```jsonl
{"question": "What is 15 + 27?", "output": "42"}
{"question": "What is the capital of France?", "output": "Paris"}
```

## API Reference

### Types

```python
from agentopt import Message, EvaluationTask, ModelResult, SelectionResults

# Message in a conversation
Message(role="user", content="Hello")

# Task for evaluation
EvaluationTask(messages=[Message(...)], expected_answer="42")

# Result from model evaluation
ModelResult(model_name="gpt-4o", accuracy=0.95, latency_seconds=1.2, is_best=True)

# Container for all results
SelectionResults(results=[...])
results.get_best()           # Get best overall
results.get_best("llm")      # Get best for specific model/attribute
results.to_csv("out.csv")    # Export to CSV
```

### ModelProxy

```python
from agentopt import ModelProxy

proxy = ModelProxy(initial_model)
proxy.set_model(new_model)    # Swap model
proxy.get_model()             # Get current model
# All other attributes/methods forward to underlying model
```

### Invokers

```python
from agentopt import InvokerProtocol, CrewInvoker, LangchainInvoker, ChainedLangchainInvoker

# Protocol interface
class InvokerProtocol(Protocol):
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]: ...

# CrewAI wrapper
CrewInvoker(crew: Crew)

# LangChain wrappers
LangchainInvoker(executor: AgentExecutor)
ChainedLangchainInvoker(executors: List[AgentExecutor])  # Sequential multi-agent
```

### ModelSelector

```python
from agentopt import ModelSelector

selector = ModelSelector(
    invoker: InvokerProtocol,           # Wrapped agent
    models: Dict[ModelProxy, List[str]], # Proxy → candidate models
    accuracy_fn: Callable[[str, str], bool],  # (expected, actual) → bool
    dataset_dir: str,                    # Path to JSONL dataset
)

results: SelectionResults = selector.select_best(evaluation_tasks=None)
```

## Current Limitations

> **This project is under active development. The following components are subject to change:**

### 1. InvokerProtocol Design (Under Discussion)

The current `InvokerProtocol` works well for frameworks with clear entrypoints:
- **CrewAI**: `crew.kickoff()` provides a clean invocation point
- **LangGraph**: Similarly has well-defined entry/exit

However, for frameworks like **LangChain** where agent construction is more flexible, the invoker pattern may require users to significantly restructure their code. We're exploring alternatives that minimize user code changes.

### 2. Dataset Loading (Placeholder Implementation)

The current `load_dataset` function is a minimal implementation:
- Requires a specific `EvaluationTask` structure
- Expects JSONL with `question`/`output` fields
- May not suit all evaluation scenarios

Future versions will support:
- Custom dataset formats
- More flexible evaluation task definitions
- Integration with standard benchmarks

### 3. ModelSelector API (Subject to Change)

Since both the invoker and dataset interfaces are evolving, the `ModelSelector` API will also change accordingly. Current usage patterns may need adjustment in future releases.

## Why ModelProxy?

Many agent frameworks capture the model in closures during agent creation:

```python
# Model is captured in closure - cannot be changed later
agent = create_agent(model=ChatOpenAI(...))
```

`ModelProxy` solves this by providing an indirection layer:

```
Agent → ModelProxy → Actual Model
                  ↑
            Can be swapped!
```

The agent holds a reference to the proxy, and we can change what the proxy points to without recreating the agent.

## License

MIT
