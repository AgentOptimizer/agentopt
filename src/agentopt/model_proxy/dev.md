# Model Proxy — Developer Guide

How `ModelProxy` enables transparent model swapping across seven LLM frameworks,
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

**Key files:** `framework_specific_implementation/crewai.py` (`build_crewai_llm`, `sync_crew_agents`, `clone_crew_agents`)

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

**Key files:** `framework_specific_implementation/langchain_compat.py` (`build_langchain_compatible_llm`, `sync_langchain_executor`)

---

### 3. LlamaIndex — "Pydantic-Validated Slot"

**How it uses the LLM:**

LlamaIndex's `FunctionAgent` uses **strict Pydantic validation** on its `llm=`
parameter. The `ModelProxy` class fails Pydantic type checking, so you cannot
pass the proxy directly to the agent constructor:

```python
initial_llm = build_llamaindex_llm("gpt-4o-mini")
# Must pass the real LLM, not the proxy:
agent = FunctionAgent(llm=initial_llm, ...)
```

Post-construction, `agent.llm` is a mutable Pydantic field, so direct assignment
works. But creating a replacement LLM requires `build_llamaindex_llm()` — a
multi-provider factory that creates the correct LlamaIndex LLM class based on the
model name, with OpenRouter as a fallback when native API keys aren't available.

**Mutation strategy — factory rebuild + direct assignment:**

```python
new_llm = build_llamaindex_llm(new_model_name)
agent.llm = new_llm
```

For parallel evaluation, entire agents are cloned via
`agent.model_copy(update={"llm": fresh_llm}, deep=False)`.

**Key files:** `framework_specific_implementation/llamaindex.py` (`build_llamaindex_llm`, `sync_llamaindex_agents`, `sync_llamaindex_workflow_agents`)

---

### 4. OpenAI Agents SDK — "Abstract Base Class Interface"

**How it uses the LLM:**

The OpenAI Agents SDK checks `isinstance(model, Model)` at runtime, where
`Model` is an ABC that requires `get_response()` and `stream_response()` async
methods:

```python
agent = Agent(name="Math QA", model=proxy, ...)
# SDK internally does: isinstance(agent.model, Model) -> must be True
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

**Key files:** `framework_specific_implementation/openai_sdk.py` (`register_openai_agents_model`, `_get_response`,
`_stream_response`, `build_openai_agents_model`)

---

### 5. Claude SDK — "Functional / No Persistent Agent"

**How it uses the LLM:**

The Claude SDK is purely functional. There's no persistent agent object — you
pass configuration options at query time:

```python
options = ClaudeAgentOptions(model="haiku")
result = await query(prompt, options)  # no agent to mutate
```

**Mutation strategy — closure capture:**

```python
proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))

def invoke_fn(input_data):
    # Closure captures proxy — model swaps affect subsequent calls
    return asyncio.run(_query_async(input_data["input"], proxy))
```

No agent-side mutation is needed. The proxy is captured in a closure, so
`proxy.set_model()` changes what the next `invoke_fn()` call sees.

**Parallel support — clone_fn:**

For parallel evaluation, `clone_fn` creates fresh `ClaudeAgentOptions` per combo,
bypassing the proxy entirely so threads don't share mutable state.

**Key files:** `examples/claude_sdk_example.py`

**Note:** The Claude SDK uses short model aliases (`"haiku"`, `"sonnet"`, `"opus"`),
not full API model IDs.

---

### 6. AG2 (AutoGen 2) — "Registration + Patching"

**How it uses the LLM:**

AG2's `ConversableAgent` accepts `llm_config=LLMConfig(config_list=...)` and
validates the type via `_validate_llm_config()` -> `LLMConfig.ensure_config()`.
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

**Cross-provider support:**

`_build_ag2_config(model_name)` detects the provider from the model name and
sets the correct `api_type` ("openai" vs "anthropic") and API key. This enables
AG2 to use both OpenAI and Anthropic models:

```python
_build_ag2_config("gpt-4o-mini")
# -> {"api_type": "openai", "model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}

_build_ag2_config("anthropic/claude-sonnet-4-20250514")
# -> {"api_type": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": ANTHROPIC_API_KEY}
```

**Key files:** `framework_specific_implementation/ag2.py` (`AG2ConfigWrapper`, `_build_ag2_config`, `register_ag2_llm_config`, `sync_ag2_agents`, `extract_ag2_content`)

---

### 7. LangGraph — "Compiled State Machine"

**How it uses the LLM:**

LangGraph graphs are compiled state machines. The LLM is captured in node
functions via closures at graph compilation time. You cannot swap the LLM on
an existing compiled graph — you must recompile with a new node function that
closes over the fresh LLM.

```python
def call_solver(state):
    return solver_llm.invoke(state["messages"])  # LLM captured in closure

graph = StateGraph(WorkflowState)
graph.add_node("solver", call_solver)
app = graph.compile()  # frozen — solver_llm reference is baked in
```

**Mutation strategy — full graph rebuild via clone_fn:**

LangGraph has no adapter in the registry — it always uses the `invoke_fn=` +
`clone_fn=` path. The `clone_fn` rebuilds the entire graph from scratch with
fresh LLM instances per combination:

```python
def clone_fn(model_map):
    fresh_solver = create_model_from_string(model_map[solver_proxy])
    fresh_reviewer = create_model_from_string(model_map[reviewer_proxy])
    fresh_graph = build_graph(fresh_solver, fresh_reviewer)
    return fresh_graph.invoke
```

**Key files:** `examples/langgraph_example.py`

---

## Why So Many Mutations?

Each framework makes different architectural choices along three axes:

| Decision | CrewAI | LangChain | LlamaIndex | OpenAI SDK | Claude SDK | AG2 | LangGraph |
|----------|--------|-----------|------------|------------|------------|-----|-----------|
| **When is the LLM bound?** | Agent init (copied) | Chain build (frozen) | Agent init (stored) | Runtime (interface) | Query time (passed) | Agent init (config ref) | Compile time (closure) |
| **Is the binding mutable?** | Yes (`agent.llm =`) | No (chain is immutable) | Yes (`agent.llm =`) | N/A (duck-typed) | N/A (no binding) | Yes (wrapper mutates) | No (recompile needed) |
| **Type validation?** | Loose | Loose | Strict Pydantic | `isinstance` ABC check | None | Rejects proxy | None |

These three axes create a matrix of incompatible strategies:

1. **Copy vs. reference** — CrewAI copies the LLM at init, so you must mutate
   the copy. Claude SDK uses a reference via closure, so no mutation is needed.

2. **Mutable vs. immutable structure** — LangChain's LCEL chains and LangGraph's
   compiled graphs are frozen and must be rebuilt entirely. CrewAI and LlamaIndex
   allow in-place attribute swaps.

3. **Type system strictness** — LlamaIndex rejects non-Pydantic-validated
   objects. OpenAI SDK requires ABC conformance. AG2 rejects anything that isn't
   an LLMConfig. A generic proxy can't satisfy all without framework-specific adapters.

4. **Model name field** — Even the attribute name for the model identifier
   varies: `.model`, `.model_name`, `.model_id` across providers (handled by
   the `MODEL_FIELDS` constant in `constants.py`).

5. **Object construction** — Building a replacement LLM differs per framework:
   `LLM(model=...)` for CrewAI, `ChatOpenAI(model=...)` for LangChain,
   `build_llamaindex_llm(...)` for LlamaIndex, `_build_ag2_config(...)` for AG2
   — each with different constructor signatures and provider-specific classes.

---

## Sync Flow on `proxy.set_model()`

```
proxy.set_model("gpt-4o")
  |
  +- 1. Parse model (string -> framework-specific LLM via build_llm())
  +- 2. Update _optmodel
  +- 3. Fire _sync_callbacks (registered via adapter.register_with_proxy)
        |
        +- CrewAI agents?       -> agent.llm = build_crewai_llm(name)
        +- LangChain executors? -> rebuild LCEL chain, replace executor.agent
        +- LlamaIndex agents?   -> agent.llm = build_llamaindex_llm(name)
        +- AG2 agents?          -> recreate LLMConfig + force client recreation
        +- OpenAI SDK?          -> no-op (delegates at call time via get_response)
        +- Claude SDK?          -> no-op (reads proxy via closure capture)
        +- LangGraph?           -> no-op (uses clone_fn for parallel, sequential mutates proxy)
```

---

## clone_fn — Parallel Support for invoke_fn-based Pipelines

When using `invoke_fn=` (instead of `agent=`) with `parallel=True`, you must
also supply `clone_fn`. This is required for frameworks not in the adapter
registry (LangGraph, Claude SDK) or for custom multi-agent chains.

```python
clone_fn: Callable[[Dict[ModelProxy, str]], Callable]
```

- Called once per model combination **serially** (cloning must not race)
- Receives `{proxy: model_name_str}` mapping for the combo
- Must return a fresh, independent `invoke_fn`-like callable with no shared mutable state
- Required because `invoke_fn` closures often close over proxy objects — each
  parallel thread needs its own LLM instances

Without `clone_fn`, `parallel=True` with `invoke_fn=` raises `RuntimeError`.

---

## Key Files

```
model_proxy/
├── proxy.py             # ModelProxy class, registration, _sync_callbacks
├── adapter.py           # FrameworkAdapter ABC + registry (get_adapter, register_adapter)
├── builders.py          # build_llm() dispatcher (detects framework, delegates)
├── constants.py         # Framework detection helpers, MODEL_FIELDS
└── framework_specific_implementation/
    ├── crewai.py        # build_crewai_llm, sync_crew_agents, clone_crew_agents, CrewAIAdapter
    ├── langchain_compat.py  # build_langchain_llm, sync_langchain_executor, LangChainAdapter
    ├── llamaindex.py    # build_llamaindex_llm, sync_llamaindex_agents, LlamaIndexAdapter
    ├── openai_sdk.py    # register_openai_agents_model, OpenAISDKAdapter
    └── ag2.py           # AG2ConfigWrapper, _build_ag2_config, register_ag2_llm_config, AG2Adapter
```
