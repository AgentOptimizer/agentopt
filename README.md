# AgentOptimizer — Revised Architecture Plan

## Context

The original plan proposed building a custom LLM proxy server (FastAPI + httpx) for model routing, token tracking, and cost monitoring. However, **LiteLLM** already provides all of this out of the box — OpenAI-compatible proxy, 100+ provider support, token/cost/latency tracking, streaming, and an admin API. Building our own would be redundant.

**New approach:** Two packages with clear responsibilities:

```
agentopt          — Offline model selection optimization (existing, with new factory API)
litellm (external) — Online proxy server for routing, tracking, and serving
```

```
Agent Code → OpenAI SDK (base_url=litellm) → LiteLLM Proxy → Actual LLM APIs
                                                  ↑
                                          tracks tokens, latency, cost
```

---

## Quickstart

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with Python 3.10+
uv venv --python 3.10

# Activate the environment
source .venv/bin/activate

# Install agentopt in editable mode
uv pip install -e .

# To run examples (installs crewai, langgraph, openai, etc.)
uv pip install -e ".[examples]"

# Install litellm proxy (diskcache needed for disk-based caching)
uv pip install "litellm[proxy]" diskcache

# Start the LiteLLM proxy (use full path to avoid PATH conflicts with other envs)
.venv/bin/litellm --config examples/litellm_config.yaml --port 4000

# In another terminal, try running examples:
export OPENAI_API_KEY=''
uv run python examples/crewai_example.py --selector brute_force --parallel
```

---

## What LiteLLM Gives Us (No Custom Code Needed)

| Feature | LiteLLM Capability |
|---------|-------------------|
| OpenAI-compatible proxy | `litellm --model gpt-4o --port 4000` |
| Multi-provider routing | 100+ LLMs (OpenAI, Anthropic, Google, etc.) |
| Token tracking | Built-in per-request token counting |
| Cost tracking | Built-in cost calculation per model |
| Latency tracking | Per-request latency metrics |
| Streaming | SSE streaming support |
| Model fallbacks | Automatic fallback chains |
| Rate limiting | Built-in rate limiting |
| Spend tracking | Per-user/per-key budget controls |
| Admin API | `/spend/logs`, `/model/info`, `/health` |
| Callbacks | Custom callbacks for logging/tracking |
| Request caching | Built-in cache (in-memory, Redis, disk, s3) — identical requests return cached responses |

---

## Package 1: `agentopt` — Offline Optimization

### What stays unchanged
- All search algorithms (brute force, hill climbing, bayesian, random, hyperband, arm elimination)
- `model_price.json`, `model_topology.py`
- `base_models.py` type system

### New factory-based API (replaces ModelProxy for multi-node optimization)

```python
from agentopt import ModelSelector

def agent_maker(candidate_model_names: Dict[str, str]):
    """
    candidate_model_names: {"planner": "gpt-4o", "solver": "gpt-4o-mini"}
    Returns: a runnable agent
    """
    planner_llm = ChatOpenAI(
        model=candidate_model_names["planner"],
        base_url="http://localhost:4000/v1",  # LiteLLM proxy
    )
    solver_llm = ChatOpenAI(
        model=candidate_model_names["solver"],
        base_url="http://localhost:4000/v1",
    )
    # ... build and return agent
    return agent

selector = ModelSelector(
    agent_fn=agent_maker,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
        "solver": ["gpt-4o", "gpt-4o-mini"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)
result = selector.select_best()
# result: {"planner": "gpt-4o", "solver": "gpt-4o-mini"}
```

### Key design points

1. **`agent_fn(candidate_model_names)`** — User provides a factory function that takes a dict of `{node_name: model_name}` and returns a runnable agent. No more ModelProxy wrapping, no framework adapters needed.

2. **Token/cost tracking via LiteLLM** — During offline eval, agents call LLMs through the LiteLLM proxy. After each evaluation run, agentopt queries LiteLLM's `/spend/logs` API to get token counts and costs. No need for our own tracking.

3. **Export routing config** — `result.export_config("config.yaml")` exports the best model mapping in LiteLLM's config format, ready to load into the proxy for production serving.

### What gets deprecated
- `ModelProxy` class (replaced by factory pattern — no need for transparent wrapping)
- All framework adapters (`crewai.py`, `langchain_compat.py`, `llamaindex.py`, `openai_sdk.py`, `ag2.py`)
- `model_proxy/builders.py`, `model_proxy/constants.py`
- Custom token tracking in ModelProxy (LiteLLM handles this)
- `EvalCache` / `NoCache` (replaced by LiteLLM's built-in request caching)

### What gets added/changed
- New `ModelSelector` API accepting `agent_fn` + `models` as `Dict[str, List[str]]` (node→candidates)
- LiteLLM spend log querying for token/cost data per eval run
- Config export in LiteLLM format

---

## Package 2: LiteLLM (External — No Custom Code)

### Setup

```bash
uv pip install litellm
```

### LiteLLM Config (`litellm_config.yaml`)

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-sonnet-4-20250514
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  store_model_in_db: true  # enables spend tracking
```

### Running the proxy

```bash
litellm --config litellm_config.yaml --port 4000
```

### Querying usage (from agentopt during offline eval)

```python
import litellm

# Option 1: Use LiteLLM's Python API directly
response = litellm.completion(model="gpt-4o", messages=[...])
print(response.usage)  # tokens
print(response._hidden_params["response_cost"])  # cost

# Option 2: Query the proxy's spend API
import requests
logs = requests.get("http://localhost:4000/spend/logs").json()
```

### Request Caching (replaces agentopt's EvalCache)

LiteLLM has built-in caching that makes agentopt's `EvalCache` redundant. During offline eval, identical (model, messages, params) requests are cached at the proxy level — no application-level caching needed.

**Config (`litellm_config.yaml`):**

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: "disk"          # "local" (in-memory), "redis", "disk", or "s3"
    ttl: 3600             # cache TTL in seconds
    disk_cache_dir: ".cache/litellm"  # for disk backend
```

**Why this is better than EvalCache:**
- Caches at the proxy level — works for any agent framework or custom code automatically
- No need to pass `cache=` param to ModelSelector
- Supports multiple backends (Redis for shared/distributed eval, disk for persistence)
- Cache key includes model + messages + params, so different model combos get separate cache entries
- Can be disabled per-request via headers if needed (`"x-litellm-cache": "false"`)

### Production serving with optimized config

After agentopt finds the best models, export and reload:

```bash
# agentopt exports optimized config
python -c "result.export_config('litellm_config_optimized.yaml')"

# Restart LiteLLM with optimized routing
litellm --config litellm_config_optimized.yaml --port 4000
```

---

## Offline → Online Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  OFFLINE (agentopt)                                     │
│                                                         │
│  ModelSelector evaluates model combos via agent_fn:     │
│    planner × [gpt-4o, gpt-4o-mini, claude-sonnet]      │
│    solver  × [gpt-4o, gpt-4o-mini]                     │
│                                                         │
│  All LLM calls go through LiteLLM proxy → tracked      │
│                                                         │
│  Output: best combo = {planner: gpt-4o, solver: mini}   │
│          → exported as litellm_config.yaml              │
└──────────────────────┬──────────────────────────────────┘
                       │  litellm_config.yaml
                       ▼
┌─────────────────────────────────────────────────────────┐
│  ONLINE (LiteLLM proxy)                                 │
│                                                         │
│  Proxy runs with optimized model config                 │
│  All traffic tracked: tokens, latency, cost             │
│                                                         │
│  Dashboard: LiteLLM UI at /ui                           │
└─────────────────────────────────────────────────────────┘
```

---

## User Experience Examples

### Example 1: LangGraph with LiteLLM

```python
from langchain_openai import ChatOpenAI
from agentopt import ModelSelector

# Step 1: Define agent factory
def agent_maker(models: Dict[str, str]):
    planner_llm = ChatOpenAI(
        model=models["planner"],
        base_url="http://localhost:4000/v1",
    )
    solver_llm = ChatOpenAI(
        model=models["solver"],
        base_url="http://localhost:4000/v1",
    )
    graph = build_graph(planner_llm, solver_llm)
    return graph.compile()

# Step 2: Run optimization
selector = ModelSelector(
    agent_fn=agent_maker,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
        "solver": ["gpt-4o", "gpt-4o-mini"],
    },
    eval_fn=lambda expected, actual: expected in actual,
    dataset=dataset,
)
result = selector.select_best()
print(result.get_best())  # {planner: "gpt-4o", solver: "gpt-4o-mini"}
```

### Example 2: CrewAI with LiteLLM

```python
from crewai import Agent, Task, Crew, LLM

def agent_maker(models: Dict[str, str]):
    researcher = Agent(
        role="Researcher",
        llm=LLM(model=models["researcher"], base_url="http://localhost:4000/v1"),
    )
    writer = Agent(
        role="Writer",
        llm=LLM(model=models["writer"], base_url="http://localhost:4000/v1"),
    )
    return Crew(agents=[researcher, writer], tasks=[...])

selector = ModelSelector(
    agent_fn=agent_maker,
    models={
        "researcher": ["gpt-4o", "gpt-4o-mini"],
        "writer": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)
result = selector.select_best()
```

### Example 3: Custom Agent (No Framework)

```python
from openai import OpenAI
from agentopt import ModelSelector

def agent_maker(models: Dict[str, str]):
    """Plain while-loop agent — no framework needed."""
    client = OpenAI(base_url="http://localhost:4000/v1")

    def run(input_data):
        # Step 1: Planner generates a plan
        plan = client.chat.completions.create(
            model=models["planner"],
            messages=[{"role": "user", "content": f"Plan how to answer: {input_data}"}],
        ).choices[0].message.content

        # Step 2: Solver executes the plan (tool-use loop)
        messages = [
            {"role": "system", "content": f"Follow this plan: {plan}"},
            {"role": "user", "content": input_data},
        ]
        for _ in range(10):  # max iterations
            response = client.chat.completions.create(
                model=models["solver"],
                messages=messages,
                tools=[...],  # your tool definitions
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                # execute tools, append results, continue loop
                messages.append(msg)
                messages.append({"role": "tool", "content": execute_tool(msg.tool_calls[0])})
            else:
                return msg.content  # done
        return messages[-1].content

    return run

selector = ModelSelector(
    agent_fn=agent_maker,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini"],
        "solver": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)
result = selector.select_best()
```

---

## Implementation Phases

### Phase 1: New factory-based ModelSelector API
- New `ModelSelector` constructor accepting `agent_fn` + `models: Dict[str, List[str]]`
- Combo generation: Cartesian product of `models` values
- For each combo: call `agent_fn(combo)` → get agent → run on dataset → evaluate
- All existing search algorithms adapted to use factory pattern
- Files: `model_selection/base.py`, all selector subclasses

### Phase 2: LiteLLM integration for tracking
- Helper to query LiteLLM spend logs after each eval run
- Config export: `result.export_config()` → LiteLLM YAML format
- Example `litellm_config.yaml` template
- Files: new `litellm_utils.py`, `model_selection/base.py`

### Phase 3: Deprecation & cleanup
- Deprecate `ModelProxy`, framework adapters (keep for backward compat with warnings)
- Update examples to use new factory API + LiteLLM
- Update README

---

## Verification

1. Start LiteLLM: `litellm --config litellm_config.yaml --port 4000`
2. Run selector: `python example.py` (agent_maker points to LiteLLM)
3. Check LiteLLM tracked usage: `curl http://localhost:4000/spend/logs`
4. Export config: `result.export_config("optimized.yaml")`
5. Restart LiteLLM with optimized config, verify routing
