# agentopt

Framework-agnostic LLM model selection for multi-agent systems. Find the best model combination for each agent role by evaluating candidates on your dataset.

## Architecture

This monorepo contains two packages:

```
agentopt/                    # repo root
├── agentproxy/              # HTTP-layer LLM call tracking (standalone)
│   ├── pyproject.toml       # depends only on httpx
│   └── src/agentproxy/
│       ├── tracker.py       # LLMTracker — context management + record storage
│       ├── interceptor.py   # httpx monkey-patching + usage extraction
│       ├── cache.py         # API-level response caching (LRU, thread-safe)
│       └── models.py        # CallRecord dataclass
├── agentopt/                # Model selection optimizer
│   ├── pyproject.toml       # depends on pydantic + agentproxy
│   └── src/agentopt/
│       ├── model_selection/ # 7 selection algorithms
│       ├── model_topology.py
│       ├── model_price.py
│       └── base_models.py
├── docs/                    # Design documents
├── examples/                # Framework integration examples
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

```python
tracker = LLMTracker()                          # cache on (default), unlimited size
tracker = LLMTracker(cache=True, cache_max_size=1000)  # cache on, max 1000 entries
tracker = LLMTracker(cache=False)               # cache off
```

| Method | Description |
|--------|-------------|
| `start()` | Install httpx patches (idempotent) |
| `stop()` | Restore original httpx (idempotent) |
| `track(data_id, combo_id, agent_id=None)` | Context manager — sets attribution for all LLM calls in scope |
| `track_agent(agent_id)` | Context manager — sets only agent_id, keeps data_id/combo_id |
| `get_records(data_id=None, combo_id=None, agent_id=None)` | Filtered list of `CallRecord` |
| `get_usage(data_id=None, combo_id=None, agent_id=None)` | `{model: (input_tokens, output_tokens)}` |
| `get_cached_latency(data_id=None, combo_id=None, agent_id=None)` | Total latency (seconds) saved by cache hits |
| `cache_enabled` (property) | Get/set whether response caching is active at runtime |
| `cache_stats` (property) | `CacheStats` with `hits`, `misses`, `hit_rate` |
| `clear_cache()` | Clear all cached responses and reset stats |
| `clear()` | Clear all recorded data |

### CallRecord fields

```python
@dataclass
class CallRecord:
    data_id:           Optional[str]    # datapoint id
    combo_id:          Optional[str]    # model combination id
    agent_id:          Optional[str]    # agent role (optional)
    model:             str
    prompt_tokens:     int
    completion_tokens: int
    latency_seconds:   float
    request_url:       str
    request_body:      Dict[str, Any]
    response_body:     Dict[str, Any]
    timestamp:         str
    cached:            bool
```

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

You can also pass prebuilt LLM instances as candidates:

```python
from langchain_openai import ChatOpenAI

selector = ModelSelector(
    agent_fn=agent_maker,  # receives actual instances in models["planner"], etc.
    models={
        "planner": [ChatOpenAI(model="gpt-4o"), ChatOpenAI(model="gpt-4o-mini")],
        "solver": [ChatOpenAI(model="gpt-4o"), ChatOpenAI(model="gpt-4o-mini")],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)
```

Custom model pricing can be provided via `model_prices`:

```python
selector = ModelSelector(
    ...,
    model_prices={
        "my-custom-model": {"input_price": 2.50, "output_price": 10.00},
    },
)
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
| `LMProposalModelSelector` | Uses a proposer LLM to shortlist combinations before evaluation |

All selectors support `select_best(parallel=True, max_concurrent=20)` for async evaluation.

### LLM proposal selector

`LMProposalModelSelector` keeps the same input contract (`agent_fn`, `models`, `dataset`, `eval_fn`) and adds a proposer stage:

1. Builds a prompt with candidate model indices per node (in order) + dataset preview.
2. Calls proposer LLM (default `gpt-4o-mini`) with a Pydantic response schema.
3. Validates/deduplicates returned combinations.
4. Adds fallback baseline and exploration combinations.
5. Evaluates only the selected subset.

```python
from agentopt import LMProposalModelSelector

selector = LMProposalModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    proposer_model="gpt-4o-mini",
    max_combinations=12,
)

results = selector.select_best(parallel=True)
print(selector.last_proposal_stats)  # includes proposer_hit and source breakdown
```

Advanced tuning is optional:

```python
from agentopt import LMProposalModelSelector, LMProposalTuning

selector = LMProposalModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    tuning=LMProposalTuning(
        objective="accuracy_then_latency",
        min_include_baselines=1,
        exploration_fraction=0.2,
        dataset_preview_size=5,
        seed=7,
    ),
)
```

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

## Examples

| Example | Framework | File |
|---------|-----------|------|
| Custom agent | Raw OpenAI SDK | `examples/custom_agent_example.py` |
| OpenAI Agents SDK | `openai-agents` | `examples/openai_sdk_example.py` |
| LangChain | `langchain` | `examples/langchain_example.py` |
| LangGraph | `langgraph` | `examples/langgraph_example.py` |
| CrewAI | `crewai` | `examples/crewai_example.py` |
| AG2 | `ag2` | `examples/ag2_example.py` |

---

## Framework compatibility

agentproxy works with any framework that uses `httpx` for HTTP calls:

| Framework | Compatible | Notes |
|-----------|-----------|-------|
| OpenAI SDK | Yes | |
| OpenAI Agents SDK | Yes | Uses `openai-agents` package |
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
