# Model Proxy — Developer Guide

How `ModelProxy` enables transparent model swapping across five LLM frameworks,
and why each framework requires its own mutation strategy.

---

## The Core Problem

`ModelProxy` wraps any LLM object and lets you call `proxy.set_model(new_model)`
to swap the underlying model at any time. The difficulty is that each framework
captures the LLM at a different point in its lifecycle, stores it differently,
and enforces different type constraints. A single universal swap mechanism is
impossible — so `ModelProxy` dispatches to framework-specific sync strategies.

---

## Framework-by-Framework Breakdown

### 1. CrewAI — "Copy on Construction"

**How it uses the LLM:**

CrewAI's `Agent` **copies** the LLM during `__init__()` via an internal
`create_llm()` call. After construction, each agent holds its own independent
LLM reference in `agent.llm`.

```python
# User writes:
llm = LLM(model="openai/gpt-4o-mini")
agent = Agent(role="Researcher", llm=llm, ...)

# Internally, CrewAI does something like:
self.llm = create_llm(llm)  # copies/rebuilds — the original reference is gone
```

**Mutation strategy — direct attribute mutation:**

```python
agent.llm = build_crewai_llm(new_model_name)
```

Because the agent holds a *copy*, swapping the proxy's inner model has no effect
unless you also reach into each registered agent and overwrite `.llm`.

**Key files:** `crewai.py` (`build_crewai_llm`, `sync_crew_agents`, `clone_crew_agents`)

---

### 2. LangChain — "Baked into Immutable Chain"

**How it uses the LLM:**

LangChain builds an LCEL (LangChain Expression Language) chain at construction
time. The LLM is **structurally embedded** in the chain topology — it's a node
in a frozen computation graph:

```python
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_tool_calling_agent(llm, tools, prompt)  # LLM baked into chain
executor = AgentExecutor(agent=agent, tools=tools)
```

You cannot swap a node inside an LCEL chain after it's built.

**Mutation strategy — full chain rebuild:**

```python
new_agent = create_tool_calling_agent(new_llm, tools, prompt)
temp = AgentExecutor(agent=new_agent, tools=tools)
executor.agent = temp.agent  # replace the internal wrapped agent
```

This is why the LangChain adapter must extract and store the tools and prompt
alongside the executor: they're required to rebuild the chain from scratch.

**Key files:** `langchain_compat.py` (`build_langchain_compatible_llm`, `sync_langchain_executor`)

---

### 3. LlamaIndex — "Pydantic-Validated Slot"

**How it uses the LLM:**

LlamaIndex's `FunctionAgent` uses **strict Pydantic validation** on its `llm=`
parameter. The `ModelProxy` class fails Pydantic type checking, so you cannot
pass the proxy directly to the agent constructor:

```python
initial_llm = OpenAI(model="gpt-4o-mini")
# Must pass the real LLM, not the proxy:
agent = FunctionAgent(llm=initial_llm, ...)
```

Post-construction, `agent.llm` is a mutable Pydantic field, so direct assignment
works. But creating a replacement LLM config object requires `model_copy()` to
preserve API keys, endpoints, and other validated fields.

**Mutation strategy — Pydantic-aware clone + direct assignment:**

```python
new_llm = original_llm.model_copy(update={"model": new_model_name})
agent.llm = new_llm
```

For parallel evaluation, entire agents are cloned via
`agent.model_copy(update={"llm": fresh_llm}, deep=False)`.

**Key files:** `llamaindex.py` (`sync_llamaindex_agents`, `_build_llamaindex_llm`)

---

### 4. OpenAI Agents SDK — "Abstract Base Class Interface"

**How it uses the LLM:**

The OpenAI Agents SDK checks `isinstance(model, Model)` at runtime, where
`Model` is an ABC that requires `get_response()` and `stream_response()` async
methods:

```python
agent = Agent(name="Math QA", model=proxy, ...)
# SDK internally does: isinstance(agent.model, Model) → must be True
```

`ModelProxy` doesn't inherit from `Model`, so it would fail the check.

**Mutation strategy — ABC virtual subclass registration + method patching:**

```python
# At import time:
Model.register(ModelProxy)  # makes isinstance() pass

# Dynamically attach required methods:
ModelProxy.get_response = _get_response    # delegates to wrapped model
ModelProxy.stream_response = _stream_response
```

The proxy becomes a duck-typed `Model` without inheriting from it. The delegate
methods resolve a fresh Model instance from the current model name on each call,
so model swaps take effect immediately.

**Key files:** `openai_sdk.py` (`register_openai_agents_model`, `_get_response`,
`_stream_response`, `build_openai_agents_model`)

---

### 5. Claude SDK — "Functional / No Persistent Agent"

**How it uses the LLM:**

The Claude SDK is purely functional. There's no persistent agent object — you
pass configuration options at query time:

```python
options = ClaudeAgentOptions(model="claude-3-5-haiku-latest")
result = await query(prompt, options)  # no agent to mutate
```

**Mutation strategy — closure capture:**

```python
proxy = ModelProxy(ClaudeAgentOptions(model="claude-3-5-haiku-latest"))

def invoke_fn(input_data):
    # Closure captures proxy — model swaps affect subsequent calls
    return asyncio.run(_query_async(input_data["input"], proxy))
```

No agent-side mutation is needed. The proxy is captured in a closure, so
`proxy.set_model()` changes what the next `invoke_fn()` call sees.

**Key files:** `examples/claude_sdk_example.py`

---

### 6. AG2 (AutoGen 2) — "Registration + Patching"

**How it uses the LLM:**

AG2's `ConversableAgent` accepts `llm_config=LLMConfig(config_list=...)` and
validates the type via `_validate_llm_config()` → `LLMConfig.ensure_config()`.
`LLMConfig` is a standalone class (not ABC, not Pydantic), so virtual subclass
registration is not possible.

**Mutation strategy — registration + patching + agent sync:**

`register_ag2_llm_config(ModelProxy)` is called at import time and applies three
patches:

1. **`ModelProxy.__init__`** — detects LLMConfig input and silently converts it
   to an internal `AG2ConfigWrapper` that has a `model` property for
   `set_model()` to update.

2. **`ConversableAgent._validate_llm_config`** — detects ModelProxy and creates
   a real `LLMConfig(config_list=wrapper.config_list)` that shares the mutable
   `config_list` by reference with the wrapper.

3. **`ConversableAgent.__init__`** — auto-registers the agent with the proxy
   when `llm_config=proxy` is passed. This allows `_sync_registered_frameworks()`
   to explicitly sync the agent on `set_model()`.

On `set_model()`, `sync_ag2_agents()` recreates the `LLMConfig` from the
wrapper's updated `config_list` and injects it into each registered agent,
also forcing client recreation if AG2 caches an `OpenAIWrapper` at init time.

This gives users a native-feeling API:

```python
llm_config = LLMConfig({"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")})
proxy = ModelProxy(llm_config)
agent = ConversableAgent(name="...", llm_config=proxy)
# Agent is auto-registered — no explicit registration call needed.
```

The selector auto-detects AG2 agents via `is_ag2_agent()` and wraps `.run()`
with `_make_ag2_invoke_fn()`, which handles the `message=` parameter and
response content extraction via `extract_ag2_content()`.

**Key files:** `model_proxy/ag2.py` (`AG2ConfigWrapper`, `register_ag2_llm_config`, `sync_ag2_agents`, `extract_ag2_content`), `examples/ag2_example.py` (agent definitions)

---

## Why So Many Mutations?

Each framework makes different architectural choices along three axes:

| Decision | CrewAI | LangChain | LlamaIndex | OpenAI SDK | Claude SDK | AG2 |
|----------|--------|-----------|------------|------------|------------|-----|
| **When is the LLM bound?** | Agent init (copied) | Chain build (frozen) | Agent init (stored) | Runtime (interface) | Query time (passed) | Agent init (config ref) |
| **Is the binding mutable?** | Yes (`agent.llm =`) | No (chain is immutable) | Yes (`agent.llm =`) | N/A (duck-typed) | N/A (no binding) | Yes (wrapper mutates) |
| **Type validation?** | Loose | Loose | Strict Pydantic | `isinstance` ABC check | None | Rejects proxy |

These three axes create a matrix of incompatible strategies:

1. **Copy vs. reference** — CrewAI copies the LLM at init, so you must mutate
   the copy. Claude SDK uses a reference via closure, so no mutation is needed.

2. **Mutable vs. immutable structure** — LangChain's LCEL chains are frozen
   computation graphs that must be rebuilt entirely. CrewAI and LlamaIndex allow
   in-place attribute swaps.

3. **Type system strictness** — LlamaIndex rejects non-Pydantic-validated
   objects. OpenAI SDK requires ABC conformance. A generic proxy can't satisfy
   both without framework-specific adapters.

4. **Model name field** — Even the attribute name for the model identifier
   varies: `.model`, `.model_name`, `.model_id` across providers (handled by
   the `MODEL_FIELDS` constant in `constants.py`).

5. **Object construction** — Building a replacement LLM differs per framework:
   `LLM(model=...)` for CrewAI, `ChatOpenAI(model=...)` for LangChain,
   `OpenAI(model=...)` for LlamaIndex — each with different constructor
   signatures and provider-specific classes.

---

## Sync Flow on `proxy.set_model()`

```
proxy.set_model("gpt-4o")
  │
  ├─ 1. Parse model (string → framework-specific LLM via build_llm())
  ├─ 2. Update _optmodel
  └─ 3. _sync_registered_frameworks()
        │
        ├─ CrewAI agents?     → agent.llm = build_crewai_llm(name)
        ├─ LangChain executors? → rebuild LCEL chain, replace executor.agent
        └─ LlamaIndex agents? → agent.llm = model_copy(update={...})
```

OpenAI SDK and Claude SDK don't need sync — OpenAI delegates at call time via
`get_response()`, and Claude SDK reads the proxy via closure capture.

---

## Key Files

```
model_proxy/
├── base.py          # ModelProxy class, registration, _sync_registered_frameworks
├── builders.py      # build_llm() dispatcher (detects framework, delegates)
├── constants.py     # Framework detection helpers, MODEL_FIELDS
├── crewai.py        # build_crewai_llm, sync_crew_agents, clone_crew_agents
├── langchain_compat.py  # build_langchain_compatible_llm, sync_langchain_executor
├── llamaindex.py    # sync_llamaindex_agents, _build_llamaindex_llm
├── openai_sdk.py    # register_openai_agents_model, _get_response, _stream_response
└── ag2.py           # AG2ConfigWrapper, register_ag2_llm_config, extract_ag2_content
```
