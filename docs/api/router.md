# Router

A `Router` is a policy that decides — **per individual LLM call** — which model to send the request to. The swap happens transparently at the HTTP layer, so it works with any framework (OpenAI SDK, Anthropic, LangChain, LangGraph, CrewAI) and any subprocess agent (Claude Code, Gemini CLI, etc.) — no integration code.

```python
from agentopt import RandomRouter

with RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"], seed=0):
    answer = agent.run(question)         # each LLM call routed independently
```

That's the whole user-facing API. The same code works whether the proxy runs in-process (default) or through a long-lived `agentopt serve` daemon (`AGENTOPT_GATEWAY_URL=…` env var). No `LLMTracker` boilerplate, no library/daemon code-path split.

---

## Why routing (vs. selection)

A [model selector](selectors.md) picks **one combination** of models and evaluates it on the whole dataset. A router is orthogonal: given a pool of candidate models, it decides at runtime, **for every LLM call**, which one to use — based on the prompt, session metadata, or prior calls in the same workflow.

| | ModelSelector | Router |
|:---|:---|:---|
| Grain | One combo per experiment | One model per call |
| Decision time | Before the run | At each HTTP request |
| Natural algorithm | UCB, arm elimination, Bayesian | Rule-based, classifier, bandit |
| Activation | `selector.select_best()` | `with router:` / `track(router=…)` |

---

## Public API

```python
from agentopt import Router, RandomRouter
```

Two names. `RouteContext` (the type of `ctx` on `Router.route`) lives at `agentopt.routing.RouteContext` — only needed if you want a type annotation; most custom routers just access fields by attribute.

### `Router`

| Member | Description |
|:---|:---|
| `route(ctx) -> Optional[str]` | **Implement this.** Return a model name to swap, or `None` to keep the client's choice. |
| `with router:` | Auto-spins an ephemeral `LLMTracker` and a tracking session so the router does something with no other setup. |

### `RouteContext`

Read-only by contract.

| Field | Type | Description |
|:---|:---|:---|
| `request_body` | `dict` | Parsed inbound JSON. |
| `provider` | `str` | `"openai"`, `"anthropic"`, `"google"`, or `"unknown"`. |
| `requested_model` | `str?` | The model the client originally asked for. |
| `session` | `SessionInfo` | Active session (`data_id`, `combo_id`, `agent_id`, records). |
| `history` | `Sequence[CallRecord]` | Snapshot of prior calls in this session. |

### `RandomRouter`

```python
RandomRouter(candidates: Sequence[str], seed: int | None = None)
```

Uniform random pick. `seed` makes choices reproducible.

---

## Writing a custom policy

Subclass `Router` and implement `route`:

```python
from agentopt import Router

class FirstCallBigRouter(Router):
    """Big model on the first call of a workflow, cheap model after."""

    def __init__(self, big: str, small: str) -> None:
        self.big = big
        self.small = small

    def route(self, ctx):
        return self.big if not ctx.history else self.small

with FirstCallBigRouter(big="gpt-4o", small="gpt-4o-mini"):
    agent.run(question)
```

- Return `None` to keep the requested model.
- Exceptions inside `route()` are caught and logged; the request proceeds unrouted. A router must never break an agent.
- One policy per file, mirroring `agentopt.model_selection` — drop new policies into `agentopt/routing/` as siblings of `random_policy.py`.

---

## Activation

Two patterns. Pick one per scope — don't nest them.

| Pattern | When to use |
|:---|:---|
| **`with router:`** (standalone) | Ad-hoc routing. Auto-spins an ephemeral tracker. The one-liner. |
| **`tracker.track(router=…)`** | You already have an `LLMTracker` and want explicit `data_id` / `combo_id` attribution for `tracker.get_records()`. |

```python
# Standalone — one line
with router:
    agent.run(question)

# Inside an explicit tracker
with tracker.track(data_id="dp_1", combo_id="gpt4o-pool", router=router):
    agent.run(question)
```

Opening `with router:` *inside* an already-active `tracker.track()` raises — the surrounding session was constructed without the router and would silently fail to apply it. The error message points you at `track(router=...)`.

---

## Daemon mode

When `AGENTOPT_GATEWAY_URL` is set, the same Python code routes via the daemon — `Router.config()` serializes the router and POSTs it on `POST /sessions`; the daemon reconstructs it from a registry and applies it in its mitmproxy addon.

### Default policy at startup

Configure a daemon-wide default that applies to every session that doesn't carry its own:

```bash
agentopt serve --routing-policy random \
    --candidate-models gpt-4o,gpt-4o-mini \
    --seed 42
```

Currently only `random` is wired to CLI flags. Pass `--candidate-models` without `--routing-policy` and the daemon refuses to start.

### Per-session override

A Python client's `with RandomRouter(...):` always overrides the daemon default for that session — no extra code. To opt out of a daemon default for a single session (run that session unrouted), use the wire protocol directly (see [proxy.md](proxy.md) for `POST /sessions`'s `router` field).

### Custom routers in daemon mode

Built-in routers work over the wire. **Custom user-defined routers are library-only in v1** — the daemon refuses to instantiate arbitrary user classes for security. Phase 3 lifts this via `agentopt serve --policy-module ./my_policies.py`; the wire protocol is already compatible.

---

## How it works

The swap happens at the same HTTP-layer seams the [proxy](proxy.md) already uses for tracking and caching:

- **In-process httpx path** — `LocalHandler` in `agentopt.proxy.interceptor`. The patched `httpx.Client.send` dispatches to the active session's handler, which applies the router and rebuilds the request with the new `body["model"]` (a fresh `httpx.Request` so its `.stream` and `Content-Length` stay coherent).
- **Subprocess path** — `AgentoptAddon.request` in `agentopt.proxy.mitm_addon`. The mitmproxy addon owned by the session reads the same router and mutates `flow.request.content`.

Both sites delegate to one shared dispatcher in `agentopt.routing.base.apply_router`:

```
agent.run(q)
  └── httpx.Client.send  →  patched send  →  active.handler.handle_sync
      └── apply_router(router, body, path, session)
          └── ctx = RouteContext(body, provider, requested_model, session, history)
          └── model_name = router.route(ctx)
          └── body["model"] = model_name  (if not None)
      └── (post-routing) cache lookup, forward to upstream, record
```

Two invariants worth knowing:

1. **Cache keys reflect the routed model, not the requested one.** Routing runs *before* `_make_cache_key`, so two identical request bodies routed to different models produce distinct cache entries. Otherwise a routed call could return a response generated by a different model.
2. **The httpx-patch install/uninstall is refcounted**, so an ephemeral router-spawned tracker and a user-constructed `LLMTracker` coexist without one tearing down the other's patch on exit.

---

## v1 limits

- **Same-provider only.** `route()` returns `Optional[str]` — a model name in the same provider as the request URL. Cross-provider routing (host + auth + schema rewrites) is on the [roadmap](#roadmap).
- **OpenAI / Anthropic body shape only.** The model lives in `request_body["model"]`. Gemini's URL-encoded model (`/v1beta/models/{model}:generateContent`) isn't rewritten yet — `route()` is still called, but a non-`None` decision logs at DEBUG and the request passes unrouted.
- **Custom routers don't work in daemon mode** (built-ins only). See above.
- **Not re-entrant on a single instance.** `with router:` on the same instance twice raises. Use distinct instances for nested or concurrent scopes.
- **No nesting `with router:` inside `tracker.track()`.** Raises with a pointer to use `track(router=...)`.

---

## Where the code lives

```
src/agentopt/routing/
├── base.py            Router, RouteContext, apply_router dispatcher, config/_config_kwargs
├── random_policy.py   RandomRouter
├── config.py          BUILTIN_POLICIES registry + resolve_policy()
└── __init__.py        public re-exports
```

Top-level re-exports: `agentopt.Router`, `agentopt.RandomRouter`.

---

## Roadmap

- **Plugin loader** — `agentopt serve --policy-module ./my_policies.py` pre-imports user modules so their `Router` subclasses resolve over the wire. Wire protocol already supports this.
- **Gemini path-rewrite routing** — extend the dispatcher to rewrite `/v1beta/models/{model}:…` URLs when the body has no `model` field.
- **Cross-provider routing** — `route() -> Optional[str | RouteDecision]`; `RouteDecision` carries provider/api_key and the dispatcher rewrites host + auth + (where required) the request schema. Streaming response translation is the hard part.
- **Selection ↔ routing loop** — feed `ModelSelector` results into a learned `Router` (bandit, classifier on prompt features) so selection and routing compound rather than merely coexist. ModelSelector doesn't accept a `router=` parameter yet.
