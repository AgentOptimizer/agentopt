# Agent Efficiency Optimization - Model Selection Package

A package for enabling model selection/hyperparameter tuning with LangChain agents while minimizing changes to user code.

## Installation

Using `uv` (recommended):

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (installs LangChain and related packages)
uv sync

# Run example
uv run python examples/main.py
```

The `uv sync` command will:
- Create a virtual environment (`.venv`)
- Install LangChain, langchain-openai, pandas, matplotlib
- Install the model_selection package in editable mode

**Note:** You'll need to set your OpenRouter API key. Create a `.env` file:
```bash
cp .env.example .env
# Then edit .env and add your OPENROUTER_API_KEY
```

Or set it as an environment variable:
```bash
export OPENROUTER_API_KEY=your_key_here
```

## How to Use Model Selector for LangChain

### Original LangChain Agent Creation

Here's how you would normally create a LangChain agent:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

# Create agent with a specific model
agent = create_agent(
    model=ChatOpenAI(
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ),
    tools=[add],
    system_prompt="You are a helpful math assistant.",
)

# Use the agent
result = agent.invoke({"messages": [{"role": "user", "content": "What is 15 + 27?"}]})
```

**The problem:** Once the agent is created, you cannot change the model. The model is captured in a closure inside the compiled graph, making it impossible to swap models for evaluation or optimization.

### Solution: Use ModelProxy

To enable model swapping, we need to modify the agent creation to use a `ModelProxy` instead of a direct model. Here's how:

### Step 1: Create Your Agent with ModelProxy

Modify your agent creation to use `ModelProxy`:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agentopt import ModelProxy
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def MyLangchainAgent(model_name: str = "openai/gpt-4o"):
    # Create initial model
    initial_model = ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    # Create a proxy model that will forward to the current model
    proxy_model = ModelProxy("model", initial_model)
    
    # Create agent with proxy model
    agent = create_agent(
        model=proxy_model,
        tools=[add],
        system_prompt="You are a helpful math assistant.",
    )
    
    # Store proxy reference on agent for later access
    agent._model = proxy_model
    
    return agent
```

### Step 2: Use ModelSelector for Model Selection

```python
from agent import MyLangchainAgent
from agentopt import ModelSelector

# Define accuracy function
def accuracy_fn(expected_answer: str, actual_output: str) -> bool:
    """Check if expected answer appears in actual output."""
    return expected_answer.lower() in actual_output.lower()

# Create agent
agent = MyLangchainAgent()

# Create model selector
selector = ModelSelector(
    agent=agent,
    dataset_dir="datasets",  # Path to JSONL dataset
    models={
        "._model": [  # Attribute path where proxy is stored
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-3-flash-preview",
        ]
    },
    accuracy_fn=accuracy_fn,
)

# Run model selection
results_df = selector.select_best()

# Results include accuracy, latency, and best model flag
print(results_df)
```

### Step 3: Dataset Format

Create a JSONL file in your dataset directory with format:

```jsonl
{"question": "What is 15 + 27?", "output": "42"}
{"question": "Solve: 8 × 7 - 12", "output": "44"}
```

Each line is a JSON object with:
- `question`: The input question/prompt
- `output`: The expected answer (used for accuracy checking)

## Why We Need Proxy Model for LangChain

### The Problem: Models Are Stored in Closures

LangChain agents are built using **LangGraph**, which creates a **compiled state graph**. When you create an agent:

```python
agent = create_agent(model=ChatOpenAI(...), ...)
```

The model is **captured in a closure** of the graph node function, not stored as an attribute:

```python
# Internally, LangChain does something like:
def create_agent(model, tools, ...):
    def model_node(state):
        # model is captured in this closure
        return model.invoke(state)
    
    graph.add_node("model", model_node)
    return graph.compile()
```

**The problem:** You can't change the model after creation because:
- `agent.model` doesn't exist as an attribute
- The model is hidden inside the closure
- Changing `agent.model` (if it existed) wouldn't affect the closure

### The Solution: Proxy Pattern

We use a `ModelProxy` that forwards calls to the current model:

```python
# Create agent with proxy
proxy_model = ModelProxy("model", initial_model)
agent = create_agent(model=proxy_model, ...)
agent._model = proxy_model  # Store reference

# Later, swap models:
from agentopt import bind_model
bind_model(agent, "_model", new_model)  # Updates proxy._model
```

**How it works:**
1. `ModelProxy` implements `__getattr__` to forward all method calls
2. When the agent's graph node calls `proxy.invoke()`, it forwards to `proxy._model`
3. We can update `proxy._model` externally without recreating the agent
4. The closure still references the proxy, but the proxy forwards to the current model

### Visual Explanation

```
┌─────────────────────────────────────┐
│   CompiledStateGraph (agent)       │
│                                     │
│   ┌──────────┐      ┌──────────┐   │
│   │  model   │─────▶│  tools   │   │
│   │  node    │      │  node    │   │
│   └──────────┘      └──────────┘   │
│        │                            │
│        │ Closure contains:         │
│        │ - proxy_model (ModelProxy)│
│        │   └─> proxy._model        │
│        │       (can be swapped!)   │
│        └──────────────────────────│
└─────────────────────────────────────┘
```

### Why Not Just Replace agent.model?

```python
# This doesn't work:
agent.model = new_model  # ❌ agent.model doesn't exist

# Even if it did:
agent.model = new_model  # ❌ Doesn't affect closure
```

The closure was created when `create_agent()` was called. Changing attributes after creation doesn't affect what's already captured in the closure.

### Benefits of Proxy Pattern

1. **Zero changes to LangChain internals** - We work with the existing architecture
2. **Dynamic model swapping** - Change models without recreating agents
3. **Minimal code changes** - Just wrap initial model with `ModelProxy`
4. **Transparent** - Proxy forwards all calls, so it behaves like a real model

## Features

- **Zero changes** to user's agent code (except wrapping model with `ModelProxy`)
- **Explicit binding** - user specifies the attribute path (e.g., "._model")
- **Nested attributes** - support for multi-level paths like `agent.B.C`
- **Simple API** - just `bind_model(obj, attr_path, model)`
- **Dataset support** - JSONL format with question/output pairs
- **Visualization** - automatic plotting of accuracy vs latency

## API Reference

### `ModelSelector`

```python
ModelSelector(
    agent: Any,
    models: Dict[str, List[str]],
    accuracy_fn: Callable[[str, str], bool],
    dataset_dir: Optional[str] = None,
)
```

- `agent`: The agent instance to optimize
- `models`: Dictionary mapping attribute paths to list of model names
- `accuracy_fn`: Function that takes (expected_answer, actual_output) and returns bool
- `dataset_dir`: Path to directory containing JSONL dataset files

Returns a pandas DataFrame with columns: `model_name`, `accuracy`, `latency_seconds`, `attribute`, `best`

### `bind_model(obj, attr_path, model)`

Bind a model to an object's attribute, supporting nested paths.

**Parameters:**
- `obj`: Object to bind model to
- `attr_path`: Path to the attribute (e.g., "._model", "B.C")
- `model`: The model object to bind

### `ModelProxy`

A proxy object that forwards attribute access to the current model.

**Usage:**
```python
proxy = ModelProxy("model", initial_model)
# Later update:
proxy._model = new_model
```

## Example

See `examples/main.py` for a complete working example that:
1. Creates an agent with `ModelProxy`
2. Evaluates multiple models on a dataset
3. Selects the best model
4. Generates a visualization
