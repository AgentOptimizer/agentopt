# `agentopt.proxy` — HTTP-Layer Context and Observability for LLM Agents

> **Core thesis:** Instead of reaching into agent frameworks to find where LLM calls are made, we patch at the HTTP layer — the one chokepoint all frameworks share. Context is propagated via Python's `contextvar` mechanism, requiring zero changes to agent code.

---

## 1. The Problem

When evaluating multi-agent systems across model combinations, you need to answer:

- What did this model combination cost on this datapoint?
- Which agent role drove the most token usage?
- How does latency break down across agent roles?

LiteLLM (or any LLM backend) handles LLM-call-level infrastructure but has no concept of which datapoint is being evaluated, which model combination is under test, or which agent role made a call. That context lives in the evaluation harness.

The naive fix — wrapping LLM objects inside the agent to inject context — requires reaching into each framework's internals to find where the LLM is stored. Every framework does this differently:

- CrewAI copies the LLM into `agent.llm` at init time
- LangGraph captures it in a compiled closure
- LangChain bakes it into an immutable LCEL chain
- AG2 hides it behind `OpenAIWrapper`

This produces one adapter per framework, each with its own mutation strategy, and imposes constraints on how agents must be designed. It's the wrong layer.

**The right layer is HTTP.** Every LLM call — regardless of framework, regardless of how many abstraction layers sit above it — eventually calls `httpx.Client.send()`. Patch that one function and you see everything.

---

## 2. Core Mechanism

### Monkey-patching `httpx.Client.send()`

When `tracker.start()` is called, agentopt.proxy replaces `httpx.Client.send` and `httpx.AsyncClient.send` at the class level:

```python
_original_send = httpx.Client.send

def _patched_send(self, request, *, stream=False, **kwargs):
    t0 = time.monotonic()
    response = _original_send(self, request, stream=stream, **kwargs)
    latency_ms = (time.monotonic() - t0) * 1000
    if not stream and response.status_code == 200:
        _try_record(request, response, latency_ms)
    return response

httpx.Client.send = _patched_send       # sync
httpx.AsyncClient.send = _patched_send  # async
```

Because this patches the **class**, not an instance, it intercepts every `httpx.Client` in the entire process — including clients created internally by CrewAI, LangGraph, LangChain, LlamaIndex, OpenAI SDK, and any other framework. They all share the same class.

The call chain for any framework looks like:

```
crew.kickoff()                  # or agent.invoke(), app.invoke(), etc.
    └── ... framework internals ...
          └── httpx.Client.send()   ← patched, we see it here
                └── LiteLLM / LLM API
```

It doesn't matter how many abstraction layers are above. Every HTTP request in `httpx` bottoms out at `send()`.

### ContextVar for attribution

A `contextvar` is a Python built-in that acts like a scoped global variable — each thread and each async task gets its own independent value. The evaluation harness sets the context once at the trajectory boundary; the patched `send()` reads it on every HTTP call below.

```python
# module level in agentopt.proxy
_current_data_id  = ContextVar("data_id",  default=None)
_current_combo_id = ContextVar("combo_id", default=None)
_current_agent_id = ContextVar("agent_id", default=None)
```

```python
# tracker.track() sets them
def __enter__(self):
    self._t1 = _current_data_id.set(self.data_id)
    self._t2 = _current_combo_id.set(self.combo_id)
    self._t3 = _current_agent_id.set(self.agent_id)

def __exit__(self, *_):
    _current_data_id.reset(self._t1)
    _current_combo_id.reset(self._t2)
    _current_agent_id.reset(self._t3)
```

```python
# _patched_send() reads them
def _try_record(request, response, latency_ms):
    data_id  = _current_data_id.get()
    combo_id = _current_combo_id.get()
    agent_id = _current_agent_id.get()
    # parse response body, write CallRecord
```

The contextvar is a module-level variable so `_patched_send` can always read it — no parameters need to be threaded through. Within the same execution context (thread or async task), the value flows down the entire call stack automatically.

### Why this works for parallel evaluation

Each thread and each `asyncio` task gets its own independent copy of every contextvar. So two parallel evaluations setting different `combo_id` values never interfere:

```
Thread A: combo_id="gpt4o+haiku"  → httpx.send() reads "gpt4o+haiku" ✅
Thread B: combo_id="mini+mini"    → httpx.send() reads "mini+mini"   ✅
```

Even though both threads call the same patched `_patched_send` function, `_current_combo_id.get()` returns a different value in each — the function is shared, the state is not.

---

## 3. Parallel Evaluation of Sync Agents

Agent code is typically synchronous. Parallel evaluation is achieved via `asyncio.to_thread`, which runs a sync function in a worker thread and — critically — **copies the current context into that thread** (Python 3.11+). This means the contextvar set by `tracker.track()` is visible inside the worker thread when `httpx.Client.send()` runs.

```python
async def evaluate_one(models, data_id, input_data, tracker):
    agent = agent_maker(models)       # fresh agent, no shared state
    combo_id = make_combo_id(models)
    with tracker.track(data_id=data_id, combo_id=combo_id):
        result = await asyncio.to_thread(agent, input_data)
    return tracker.get_usage(data_id=data_id, combo_id=combo_id), result

async def evaluate_all(candidate_combinations, input_data, tracker):
    tasks = [
        evaluate_one(models, f"dp_{i}", input_data, tracker)
        for i, models in enumerate(candidate_combinations)
    ]
    return await asyncio.gather(*tasks)
```

Threading picture:

```
Main async task
  ├── tracker.track(combo_id="gpt4o+haiku")
  │     └── to_thread(agent_A, input)     # worker thread, context copied
  │           └── httpx.Client.send()     # reads combo_id="gpt4o+haiku" ✅
  │
  └── tracker.track(combo_id="mini+mini")
        └── to_thread(agent_B, input)     # different worker thread
              └── httpx.Client.send()     # reads combo_id="mini+mini"   ✅
```

Each `agent_maker` call produces a fresh agent with no shared state, so there are no race conditions between parallel runs.

---

## 4. Context Model

Three IDs form the attribution key for every LLM call:

| ID | Meaning | Required? |
|----|---------|-----------|
| `data_id` | Which datapoint is being evaluated | Yes |
| `combo_id` | Which model combination is under test | Yes |
| `agent_id` | Which agent role made this call | No — optional |

`data_id` and `combo_id` are always set at the trajectory boundary by the eval harness. `agent_id` is optional — it requires wrapping at the individual agent node level and is only needed for per-agent attribution within a trajectory.

```python
# Trajectory-level (primary use case)
# One with() at the top — all LLM calls inside inherit the context
with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku"):
    result = agent(input)

# Agent-level (optional, for finer attribution)
# Wrap individual agent invocations inside agent_maker
def planner_node(state):
    with tracker.track_agent("planner"):   # only flips agent_id, keeps data_id/combo_id
        response = planner_llm.invoke(...)
    return {"plan": response.content}
```

All three IDs are stored on every `CallRecord`, enabling aggregation at any granularity:

```python
# Total trajectory cost
tracker.get_usage(data_id="dp_1", combo_id="gpt4o+haiku")

# Per-agent cost (requires agent_id to have been set)
tracker.get_usage(data_id="dp_1", combo_id="gpt4o+haiku", agent_id="planner")

# Per-model cost across all trajectories
tracker.get_usage(combo_id="gpt4o+haiku")
```

---

## 5. Preconditions

**The only requirement: LLM calls must go through `httpx`.**

All six target frameworks satisfy this — they all use `httpx` under the hood.

**The contextvar requirement: LLM calls must happen in a context descended from the caller.**

This is automatically satisfied for:
- Same-thread synchronous calls
- `asyncio` tasks (each task gets its own context copy)
- `asyncio.to_thread` (explicitly copies context into the worker thread)

This breaks when a framework spawns a **new thread without propagating context**. The contextvar is invisible in that thread and attribution produces records with all IDs set to `None`.

### Framework compatibility

| Framework | HTTP library | Execution model | Compatible? | Notes |
|-----------|-------------|----------------|-------------|-------|
| LangChain | httpx | Sync, caller's thread | ✅ | |
| LangGraph | httpx | Sync, caller's thread | ✅ | |
| CrewAI | httpx | Sync, caller's thread | ✅ | |
| LlamaIndex | httpx | Sync, caller's thread | ✅ | |
| OpenAI SDK | httpx | Sync, caller's thread | ✅ | |
| Custom (raw openai/httpx) | httpx | Sync, caller's thread | ✅ | |
| AG2 | httpx | `run()` spawns background thread | ⚠️ | Use `initiate_chat()` |

### AG2 constraint

AG2's `run()` API starts the chat in a background thread without propagating context. Use `initiate_chat()` instead, which blocks in the caller's thread.

```python
# ❌ run() spawns background thread — contextvar lost
with tracker.track(data_id="dp_1", combo_id="..."):
    agent.run(recipient, message=input).process()

# ✅ initiate_chat() blocks in caller's thread — contextvar propagates
with tracker.track(data_id="dp_1", combo_id="..."):
    user_proxy.initiate_chat(agent, message=input)
```

---

## 6. API Design

### `LLMTracker`

```python
from agentopt.proxy import LLMTracker

tracker = LLMTracker()
tracker.start()   # installs httpx patch

with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku"):
    result = agent(input_data)

# Query
usage   = tracker.get_usage(data_id="dp_1", combo_id="gpt4o+haiku")
# → {"gpt-4o": {"input_tokens": 500, "output_tokens": 200}, ...}

records = tracker.get_records(data_id="dp_1", combo_id="gpt4o+haiku")

tracker.stop()    # restores original httpx
```

| Method | Description |
|--------|-------------|
| `start()` | Install httpx patch. Call once before any tracking. |
| `stop()` | Uninstall patch, restore original httpx. |
| `track(data_id, combo_id, agent_id=None)` | Context manager. Sets all IDs for the duration of the block. |
| `track_agent(agent_id)` | Context manager. Flips only `agent_id`, keeps `data_id` and `combo_id`. For use inside agent nodes. |
| `get_records(data_id=None, combo_id=None, agent_id=None)` | Return `CallRecord` list, filtered by any combination of IDs. |
| `get_usage(data_id=None, combo_id=None, agent_id=None)` | Return aggregated `{model: (input_tokens, output_tokens)}`. |
| `get_cached_latency(data_id=None, combo_id=None, agent_id=None)` | Total latency (seconds) saved by cached responses. |
| `cache_enabled` (property) | Get/set whether response caching is active at runtime. |
| `clear_cache()` | Clear all cached responses and reset statistics. |
| `clear()` | Clear all recorded data. |

### `CallRecord`

```python
@dataclass
class CallRecord:
    # Attribution (from ContextVars)
    data_id:   Optional[str]   # datapoint id
    combo_id:  Optional[str]   # model combination id
    agent_id:  Optional[str]   # agent role (optional)

    # LLM call metrics
    model:              str
    prompt_tokens:      int
    completion_tokens:  int
    latency_seconds:    float

    # Full fidelity
    request_url:   str
    request_body:  Dict[str, Any]   # full prompt sent
    response_body: Dict[str, Any]   # full response received
    timestamp:     str = ""
    cached:        bool = False
```

---

## 7. Package Structure

```
src/agentopt/proxy/
├── __init__.py        # Public API: LLMTracker, CallRecord, ResponseCache
├── tracker.py         # LLMTracker — context management + record storage + cache control
├── interceptor.py     # httpx monkey-patching + usage extraction + cache integration
├── cache.py           # ResponseCache, CacheEntry, _make_cache_key
└── models.py          # CallRecord dataclass
```

---

## 8. Capabilities

### Available now — observation

- **Token tracking** — input/output tokens per call, aggregatable by any combination of `(data_id, combo_id, agent_id, model)`.
- **Cost attribution** — estimated cost from token counts × price table, per trajectory or per agent role.
- **Latency measurement** — wall-clock time per LLM call, scoped to trajectory.
- **Full request/response logging** — complete capture of prompts and completions, scoped to trajectory.
- **API-level response caching** — SHA256 hash of `(model, messages, ...)` as cache key; return cached response on hit. Thread-safe LRU cache with configurable max size. See `docs/cache-design.md`.
- **Request deduplication** — within a trajectory, identical requests return cached responses without hitting the API.

### Available via interception — request control (future)

- **Rate limiting** — queue outgoing requests when over a per-model rate limit.
- **Retry / fallback** — on non-200 response, retry or substitute a fallback model transparently.

### Out of scope

- **Runtime model swapping** — requires a reference to the LLM object inside the agent. `agent_maker` owns that; `agentopt.proxy` does not.
- **Streaming response tracking** — no complete body at `send()` return time. Future work.
- **`requests` library** — only `httpx` is patched. Can be added with the same pattern if needed.

---

## 9. What agentopt.proxy Does Not Do (and Why)

The previous `ModelProxy` design attempted to control the backend by wrapping LLM objects and swapping models at runtime. This required one `FrameworkAdapter` per framework, each with its own mutation strategy, and imposed structural constraints on agent design.

For the target use case — **offline model selection by agent-token-cost** — runtime model swapping is unnecessary. Model selection happens by calling `agent_maker(models)` with different combinations; each combination produces a fresh agent. The combination is the atomic unit of selection and the trajectory is the atomic unit of cost measurement.

The tradeoff:

| | agentopt.proxy (current) | ModelProxy (old) |
|---|---|---|
| Token tracking | ✅ | ✅ |
| Cost attribution | ✅ trajectory-level | ✅ per-agent level |
| Latency tracking | ✅ | ❌ |
| Caching | ✅ | ❌ |
| Full request logging | ✅ | ❌ |
| Runtime model swap | ❌ | ✅ |
| Framework adapters | ✅ zero | ❌ one per framework |
| Agent design constraints | ✅ none | ❌ must expose LLM object |