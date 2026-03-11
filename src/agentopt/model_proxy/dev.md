# Model Proxy — Developer Guide

How `ModelProxy` enables transparent model swapping across seven LLM frameworks,
and why each framework requires its own mutation strategy.

---

## The Core Problem

`ModelProxy` wraps any LLM object and lets you call `proxy.set_model(new_model)`
to swap the underlying model at any time. The difficulty is that each framework
captures the LLM at a different point in its lifecycle, stores it differently,
and enforces different type constraints. A single universal swap mechanism is
impossible — so `ModelProxy` dispatches to framework-specific sync strategies
via `FrameworkAdapter` subclasses.

---

## Framework-by-Framework Breakdown

### 1. CrewAI — "Copy on Construction"

**How it uses the LLM:**

CrewAI's `Agent` **copies** the LLM during `__init__()` via an internal
`create_llm()` call. After construction, each agent holds its own independent
LLM reference in `agent.llm`.

```python
llm = LLM(model="openai/gpt-4o-mini")
agent = Agent(role="Researcher", llm=llm, ...)
# Internally: self.llm = create_llm(llm)  — the original reference is gone
```

**Mutation strategy — direct attribute mutation:**

```python
agent.llm = build_crewai_llm(new_model_name)
```

**Token tracking:** Delta-tracking via `model.get_token_usage_summary()` before
and after each call. The delta is fed into the `TokenAccumulator`.

**Parallel:** Clone crew + sub-agents with fresh LLMs wrapped in ModelProxy.
Tracker stored in `_clone_trackers[id(cloned_crew)]`.

**Key files:** `crewai.py` (`CrewAIAdapter`, `build_crewai_llm`)

---

### 2. LangChain — "Baked into Immutable Chain"

**How it uses the LLM:**

LangChain builds an LCEL chain at construction time. The LLM is **structurally
embedded** in the chain — `create_tool_calling_agent()` calls `llm.bind_tools()`,
which returns a `RunnableBinding` that's baked into the frozen chain. You cannot
swap a node inside an LCEL chain after it's built.

```python
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_tool_calling_agent(llm, tools, prompt)  # llm.bind_tools() called here
executor = AgentExecutor(agent=agent, tools=tools)
```

**Mutation strategy — full chain rebuild:**

```python
new_agent = create_tool_calling_agent(new_llm, tools, prompt)
temp = AgentExecutor(agent=new_agent, tools=tools)
executor.agent = temp.agent  # replace the internal wrapped agent
```

**Token tracking:** `_TokenTrackingCallback` (a LangChain `BaseCallbackHandler`)
is passed via `config={"callbacks": [...]}` on `agent.invoke()`. The callback's
`on_llm_end` extracts `usage_metadata` from `ChatGeneration` responses and feeds
into the `TokenAccumulator`.

Note: Proxy-level interception (`_wrap_for_usage`) does NOT work for LangChain
because `bind_tools()` returns a non-proxy `RunnableBinding` — the LCEL chain
invokes that directly, bypassing the proxy.

**Parallel:** Clone by rebuilding the entire LCEL chain with a fresh LLM wrapped
in a new `ModelProxy` with a pre-attached tracker. The `_pending_callback` bridge
connects `create_token_tracker()` to `get_invoke_fn()`.

**Key files:** `langchain_compat.py` (`LangChainAdapter`, `_TokenTrackingCallback`,
`build_langchain_compatible_llm`, `_sync_executor`)

---

### 3. LlamaIndex — "Pydantic-Validated Slot"

**How it uses the LLM:**

LlamaIndex's `FunctionAgent` uses **strict Pydantic validation** on its `llm=`
parameter. `ModelProxy` fails type checking, so you cannot pass the proxy
directly to the agent constructor.

Post-construction, `agent.llm` is a mutable Pydantic field, so direct assignment
works. But creating a replacement LLM requires `build_llamaindex_llm()` — a
multi-provider factory.

**Mutation strategy — factory rebuild + direct assignment:**

```python
new_llm = build_llamaindex_llm(new_model_name)
agent.llm = new_llm
```

**Token tracking:** Global `TokenCountingHandler` installed via
`Settings.callback_manager`. Tokens are flushed from the handler to the
`TokenAccumulator` after each `agent.run()` call in `get_invoke_fn`.

**Parallel:** Clone agents via `model_copy(update={"llm": fresh_llm}, deep=False)`.
For `AgentWorkflow`, the entire workflow is reconstructed from cloned sub-agents.

**Key files:** `llamaindex.py` (`LlamaIndexAdapter`, `build_llamaindex_llm`)

---

### 4. OpenAI Agents SDK — "Abstract Base Class Interface"

**How it uses the LLM:**

The OpenAI Agents SDK checks `isinstance(model, Model)` at runtime, where
`Model` is an ABC requiring `get_response()` and `stream_response()` async
methods.

**Mutation strategy — ABC virtual subclass registration + method patching:**

```python
# At import time (patch_proxy_class):
Model.register(ModelProxy)  # makes isinstance() pass

# Dynamically attach required methods:
ModelProxy.get_response = _get_response    # delegates to wrapped model
ModelProxy.stream_response = _stream_response
```

The delegate methods resolve a fresh Model instance from the current model name
on each call, so model swaps take effect immediately.

**Token tracking:** `_get_response` calls `extract_usage()` on the response
and feeds into the proxy's effective tracker.

**Parallel:** Clone agent with fresh ModelProxy wrapping a fresh model. Tracker
stored in `_clone_trackers[id(agent_copy)]`.

**Key files:** `openai_sdk.py` (`OpenAISDKAdapter`, `build_openai_agents_model`,
`_get_response`, `_stream_response`)

---

### 5. Claude SDK — "Functional / No Persistent Agent"

**How it uses the LLM:**

The Claude SDK is purely functional — no persistent agent object. Configuration
is passed at query time.

**Mutation strategy — closure capture:**

```python
proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))

def invoke_fn(input_data):
    return asyncio.run(_query_async(input_data["input"], proxy))
```

No agent-side mutation needed. The proxy is captured in a closure, so
`proxy.set_model()` changes what the next `invoke_fn()` call sees.

**Parallel:** Thread-local model overrides on the proxy.

**Key files:** `examples/claude_sdk_example.py`

---

### 6. AG2 (AutoGen 2) — "Intercepting Proxy via ProxyAwareWrapper"

**How it uses the LLM:**

AG2's `ConversableAgent` accepts `llm_config=LLMConfig(config_list=...)` and
builds an `OpenAIWrapper` client that makes the actual LLM API calls. The call
chain is:

```
agent.run() → generate_oai_reply() → agent.client.create(params) → OpenAI/Anthropic API
```

`ModelProxy` cannot be passed directly (AG2 validates and rejects non-LLMConfig
objects). The proxy serves as a config container (`AG2ConfigWrapper`) while
`ProxyAwareWrapper` replaces `agent.client` to intercept all LLM calls.

**Mutation strategy — three patches applied at import time:**

1. **`ModelProxy.__init__`** — detects LLMConfig input, converts to
   `AG2ConfigWrapper` (has a `model` property for `set_model()` to update).

2. **`ConversableAgent._validate_llm_config`** — detects ModelProxy, creates
   a real `LLMConfig` from the wrapper's config_list.

3. **`ConversableAgent.__init__`** — auto-registers agents with the proxy and
   replaces `agent.client` with `ProxyAwareWrapper(proxy)`.

**ProxyAwareWrapper** — the core of AG2 integration:

```
agent.run() → agent.client.create()  [client is ProxyAwareWrapper]
  → _get_effective_model_and_tracker()  [prefers instance override]
  → _build_inner_from(wrapper)          [create/cache OpenAIWrapper for current model]
  → inner.create(**config)              [actual API call]
  → tracker.add(in_tok, out_tok)        [feed token usage]
```

On `set_model()`, `_sync_callbacks` update `agent.llm_config` for consistency.
The `ProxyAwareWrapper` auto-resolves the new model on the next `create()` call.

**AG2 internal threading problem:** AG2's `agent.run()` spawns an internal
thread via `initiate_chat()`, so LLM calls happen in a *different* thread from
the one that set the proxy's thread-local override. Pure `threading.local()`
doesn't work. Solution: `ProxyAwareWrapper` has instance-level `_override_model`
and `_override_tracker` attributes, propagated by `_propagate_ag2_override()`
from `_set_thread_model()`.

**AG2 parallel serialization:** Since `_override_model`/`_override_tracker` are
instance-level (not thread-local), concurrent threads sharing the same
`ProxyAwareWrapper` would race. A per-proxy `_ag2_eval_lock` serializes the
set-override → evaluate → clear-override sequence for AG2 agents.

**Token tracking for cloned agents:** `_TrackingClientWrapper` wraps the cloned
agent's bare `OpenAIWrapper` to feed `actual_usage_summary` into the tracker
after each `create()` call.

**Cross-provider support:** `AG2ConfigWrapper._build_config(model_name)` detects
provider from name and sets `api_type` ("openai" vs "anthropic") and API key.

**Key files:** `ag2.py` (`AG2ConfigWrapper`, `ProxyAwareWrapper`,
`_TrackingClientWrapper`, `AG2Adapter`)

---

### 7. LangGraph — "Compiled State Machine"

**How it uses the LLM:**

LangGraph graphs are compiled state machines. The LLM is captured in node
functions via closures. You cannot swap the LLM on an existing compiled graph.

```python
def call_solver(state):
    return solver_llm.invoke(state["messages"])  # LLM captured in closure

graph = StateGraph(WorkflowState)
graph.add_node("solver", call_solver)
app = graph.compile()  # frozen — solver_llm reference is baked in
```

**Mutation strategy — thread-local model overrides:**

If `solver_llm` is a `ModelProxy`, `proxy.set_model()` updates the underlying
model and the closure sees the new model on next call. For parallel evaluation,
each thread sets a per-thread model override via `_set_thread_model()`.

**Key files:** `examples/langgraph_example.py`

---

## Token Tracking Architecture

All frameworks feed token counts into a unified `TokenAccumulator` (thread-safe,
resettable counter). Three extraction patterns exist:

| Pattern | Frameworks | Mechanism |
|---------|-----------|-----------|
| **Proxy-level interception** | OpenAI SDK | `_wrap_for_usage()` calls `extract_usage(response)` on return values |
| **Framework callbacks** | LangChain, LlamaIndex | `_TokenTrackingCallback.on_llm_end()`, `TokenCountingHandler` |
| **Delta tracking** | CrewAI | Before/after `get_token_usage_summary()` |
| **Client wrapper** | AG2 | `ProxyAwareWrapper.create()` reads `actual_usage_summary` |
| **Clone wrapper** | AG2 (parallel) | `_TrackingClientWrapper.create()` wraps bare `OpenAIWrapper` |

`extract_usage()` in `token_tracking.py` recognizes:
- LangChain `AIMessage.usage_metadata` (dict with `input_tokens`, `output_tokens`)
- OpenAI SDK `ModelResponse.usage` (object with `input_tokens`, `output_tokens`)
- Generic `response.usage` dict with `prompt_tokens`/`completion_tokens`

---

## Sync Flow on `proxy.set_model()`

```
proxy.set_model("gpt-4o")
  │
  ├─ 1. Parse model (string → framework-specific LLM via build_llm())
  ├─ 2. Update _optmodel
  └─ 3. Fire _sync_callbacks (registered via adapter.register_with_proxy)
        │
        ├─ CrewAI agents?       → agent.llm = build_crewai_llm(name)
        ├─ LangChain executors? → rebuild LCEL chain, replace executor.agent
        ├─ LlamaIndex agents?   → agent.llm = build_llamaindex_llm(name)
        ├─ AG2 agents?          → recreate LLMConfig (ProxyAwareWrapper auto-resolves)
        ├─ OpenAI SDK?          → no-op (delegates at call time via get_response)
        ├─ Claude SDK?          → no-op (reads proxy via closure capture)
        └─ LangGraph?           → no-op (thread-local overrides handle parallel)
```

---

## Thread-Local Model Overrides — Parallel Evaluation

When using `invoke_fn=` with `parallel=True`, `ModelProxy` uses thread-local
storage to provide each evaluation thread with its own model instance.

**How it works:**

1. Each parallel thread calls `proxy._set_thread_model(fresh_model, tracker)`.
2. `proxy._get_effective_model()` returns the thread-local model if set,
   otherwise the default. All proxy methods use this.
3. After evaluation, `proxy._clear_thread_model()` removes the overrides.

**AG2 special handling:** Because AG2 spawns internal threads for LLM calls,
thread-local alone is insufficient:

- `_set_thread_model()` also calls `_propagate_ag2_override()`, which sets
  instance-level `_override_model`/`_override_tracker` on each agent's
  `ProxyAwareWrapper` client.
- `_set_thread_model()` acquires `_ag2_eval_lock` before setting overrides.
  `_clear_thread_model()` releases it after clearing. This serializes parallel
  AG2 evaluations to prevent race conditions on the shared wrapper instances.
- For non-AG2 frameworks, the lock doesn't exist, so there's zero overhead.

---

## Why So Many Mutations?

Each framework makes different architectural choices along three axes:

| Decision | CrewAI | LangChain | LlamaIndex | OpenAI SDK | Claude SDK | AG2 | LangGraph |
|----------|--------|-----------|------------|------------|------------|-----|-----------|
| **When is the LLM bound?** | Agent init (copied) | Chain build (frozen) | Agent init (stored) | Runtime (interface) | Query time (passed) | Agent init (config ref) | Compile time (closure) |
| **Is the binding mutable?** | Yes (`agent.llm =`) | No (chain is immutable) | Yes (`agent.llm =`) | N/A (duck-typed) | N/A (no binding) | Yes (wrapper mutates) | No (recompile needed) |
| **Type validation?** | Loose | Loose | Strict Pydantic | `isinstance` ABC check | None | Rejects proxy | None |

---

## Key Files

```
model_proxy/
├── proxy.py             # ModelProxy class, thread-local overrides, AG2 lock
├── adapter.py           # FrameworkAdapter ABC + registry (get_adapter, register_adapter)
├── builders.py          # build_llm() dispatcher (detects framework, delegates)
├── constants.py         # Framework detection helpers, MODEL_FIELDS
├── token_tracking.py    # TokenAccumulator, extract_usage()
└── framework_specific_implementation/
    ├── crewai.py        # CrewAIAdapter, build_crewai_llm
    ├── langchain_compat.py  # LangChainAdapter, _TokenTrackingCallback, build_langchain_compatible_llm
    ├── llamaindex.py    # LlamaIndexAdapter, build_llamaindex_llm
    ├── openai_sdk.py    # OpenAISDKAdapter, build_openai_agents_model
    └── ag2.py           # AG2Adapter, AG2ConfigWrapper, ProxyAwareWrapper, _TrackingClientWrapper
```
