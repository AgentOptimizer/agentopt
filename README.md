# agentopt

Framework-agnostic LLM model selection for multi-agent systems. Find the best model combination for each agent role by evaluating candidates on your dataset.

## Architecture

This monorepo contains two packages:

```
agentopt/                    # repo root
├── agentproxy/              # HTTP-layer LLM call tracking (standalone)
│   ├── pyproject.toml       # depends only on httpx
│   └── src/agentproxy/
├── agentopt/                # Model selection optimizer
│   ├── pyproject.toml       # depends on pydantic + agentproxy
│   └── src/agentopt/
├── examples/
└── model_price.json
```

**agentproxy** intercepts all LLM calls at the `httpx` layer — the one chokepoint every LLM SDK shares. No proxy server, no framework adapters, no code changes to your agents.

**agentopt** uses agentproxy to measure token usage and latency while evaluating model combinations across agent roles.

---

## Quickstart

```bash
# Install both packages
uv pip install -e agentproxy
uv pip install -e agentopt

# Set your API key
export OPENAI_API_KEY='...'

# Run an example
python examples/custom_agent_example.py
```

---

## agentproxy — LLM Call Tracking

agentproxy can be used standalone for observability, independent of model selection.

```python
from agentproxy import LLMTracker

tracker = LLMTracker()
tracker.start()   # patches httpx.Client.send at the class level

with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku"):
    result = agent(input_data)  # any LLM call via any framework

usage = tracker.get_usage(data_id="dp_1", combo_id="gpt4o+haiku")
# {"gpt-4o": (500, 200), "claude-haiku": (300, 150)}
#            (input_tok, output_tok)

records = tracker.get_records(data_id="dp_1")
# List[CallRecord] with model, tokens, latency, full request/response

tracker.stop()    # restores original httpx
```

### How it works

Every LLM SDK (OpenAI, Anthropic, LangChain, CrewAI, etc.) uses `httpx` under the hood. agentproxy patches `httpx.Client.send()` and `httpx.AsyncClient.send()` at the class level, so it sees every LLM call in the process:

```
your_agent()
  └── framework internals
        └── httpx.Client.send()   ← patched, tracked here
              └── LLM API
```

Attribution is handled via Python's `contextvars` — each thread and async task gets its own independent context, so parallel evaluations never interfere.

### LLMTracker API

| Method | Description |
|--------|-------------|
| `start()` | Install httpx patches (idempotent) |
| `stop()` | Restore original httpx (idempotent) |
| `track(data_id, combo_id, agent_id=None)` | Context manager — sets attribution for all LLM calls in scope |
| `track_agent(agent_id)` | Context manager — sets only agent_id, keeps data_id/combo_id |
| `get_records(data_id=None, combo_id=None, agent_id=None)` | Filtered list of `CallRecord` |
| `get_usage(data_id=None, combo_id=None, agent_id=None)` | `{model: (input_tokens, output_tokens)}` |
| `clear()` | Clear all recorded data |

---

## agentopt — Model Selection

Define a factory function that builds your agent for a given model combination. agentopt evaluates all (or a subset of) combinations on your dataset and finds the best one.

```python
from openai import OpenAI
from agentopt import ModelSelector

client = OpenAI()

def agent_maker(models):
    def run(input_data):
        plan = client.chat.completions.create(
            model=models["planner"],
            messages=[{"role": "user", "content": f"Plan: {input_data}"}],
        ).choices[0].message.content

        answer = client.chat.completions.create(
            model=models["solver"],
            messages=[
                {"role": "system", "content": f"Follow this plan:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer
    return run

def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0

dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
]

selector = ModelSelector(
    agent_fn=agent_maker,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini"],
        "solver": ["gpt-4o", "gpt-4o-mini"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)

results = selector.select_best()
results.print_summary()
best = results.get_best_combo()
# {"planner": "gpt-4o", "solver": "gpt-4o-mini"}
```

### Selection algorithms

| Selector | Description |
|----------|-------------|
| `BruteForceModelSelector` | Evaluates all combinations (default) |
| `RandomSearchModelSelector` | Samples a fraction of combinations |
| `HillClimbingModelSelector` | Greedy search with topology-guided neighbors |
| `ArmEliminationModelSelector` | Successive elimination via statistical dominance |
| `HyperbandModelSelector` | Multi-bracket successive halving |
| `BayesianOptimizationModelSelector` | GP-based optimization (requires `torch`, `botorch`) |

All selectors support `select_best(parallel=True, max_concurrent=20)` for async evaluation.

### Results API

```python
results = selector.select_best()

results.print_summary()           # formatted table with rank, accuracy, latency, price
best = results.get_best()         # ModelResult with highest accuracy
combo = results.get_best_combo()  # {"planner": "gpt-4o", "solver": "gpt-4o-mini"}
results.to_csv("results.csv")    # export all results
results.export_config("config.yaml")  # export best combo as YAML
```

---

## Framework compatibility

agentproxy works with any framework that uses `httpx` for HTTP calls:

| Framework | Compatible | Notes |
|-----------|-----------|-------|
| OpenAI SDK | Yes | |
| LangChain / LangGraph | Yes | |
| CrewAI | Yes | |
| LlamaIndex | Yes | |
| Anthropic SDK | Yes | |
| AG2 | Partial | Use `initiate_chat()`, not `run()` |

---

## Installation

```bash
# Core packages
uv pip install -e agentproxy
uv pip install -e agentopt

# With Bayesian optimization
uv pip install -e "agentopt[bayesian]"

# With example dependencies
uv pip install -e "agentopt[examples]"
```
