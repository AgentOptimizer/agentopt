# Router

A `Router` is a policy that decides — **per individual LLM call** — which model to actually send the request to. The swap happens transparently at the HTTP layer through the same [proxy](proxy.md) that powers `LLMTracker`, so it works with any framework (OpenAI SDK, Anthropic, LangChain, LangGraph, CrewAI, subprocess agents) without any integration code.

```python
from agentopt import RandomRouter

router = RandomRouter(model_candidates=["gpt-4o", "gpt-4o-mini"])
with router:
    answer = agent.run(question)        # each LLM call routed independently
```

That's the whole public API for ad-hoc routing: construct a policy, open a `with` block, call your agent. No `LLMTracker`, no session IDs, no framework hooks.

---

## Why routing (vs. selection)

A [model selector](selectors.md) picks **one combination** of models and evaluates it on the whole dataset. A router is orthogonal: given a pool of candidate models, it decides at runtime, **for every LLM call**, which one to use — based on the prompt, session metadata, or prior calls in the same workflow.

| | ModelSelector | Router |
|:---|:---|:---|
| Grain | One combo per experiment | One model per call |
| Decision time | Before the run | At each HTTP request |
| Natural algorithm | UCB, arm elimination, Bayesian | Rule-based, classifier, bandit |
| Activation | `selector.select_best()` | `with router:` |

They compose — a future `ModelSelector` can tune a router's parameters the same way it tunes combos today.

---

## Public API

```python
from agentopt import Router, RouteContext, RouteDecision, RandomRouter
```

### `Router` (base class)

| Member | Description |
|:---|:---|
| `route(ctx: RouteContext) -> RouteDecision \| None` | **Implement this.** Return a decision to swap the model, or `None` to keep what the client asked for. |
| `__enter__` / `__exit__` | Context manager. Activates the policy for every LLM call inside the block. Not re-entrant on a single instance. |

### `RouteContext` (passed to `route`)

| Field | Type | Description |
|:---|:---|:---|
| `request_body` | `dict` | Parsed inbound JSON. Read-only by contract. |
| `provider` | `str` | `"openai"`, `"anthropic"`, `"google"`, or `"unknown"`. Derived from path. |
| `requested_model` | `str?` | The model the client asked for. |
| `session_data_id` | `str?` | From `tracker.track(data_id=…)`, if running inside a tracker. |
| `session_combo_id` | `str?` | From `tracker.track(combo_id=…)`. |
| `session_agent_id` | `str?` | From `tracker.track(agent_id=…)`. |
| `history` | `Sequence[CallRecord]` | Prior LLM calls in this session, chronological. |

### `RouteDecision` (returned by `route`)

| Field | Type | Description |
|:---|:---|:---|
| `model` | `str` | The model to actually use. |
| `provider` | `str?` | *Reserved for cross-provider routing — raises `NotImplementedError` in v1.* |
| `api_key` | `str?` | *Reserved for cross-provider routing — raises `NotImplementedError` in v1.* |

### `RandomRouter` (baseline policy)

```python
RandomRouter(model_candidates: Sequence[str], seed: int | None = None)
```

Uniform random pick from a fixed pool. `seed` makes choices deterministic for reproducible evaluation runs.

---

## Writing a custom policy

Subclass `Router` and implement `route`. The instance is a context manager by inheritance — no extra boilerplate.

```python
from agentopt import Router, RouteContext, RouteDecision

class FirstCallBigRouter(Router):
    """Big model for the first call of a workflow, cheap model afterwards."""

    def __init__(self, big: str, small: str) -> None:
        self.big = big
        self.small = small

    def route(self, ctx: RouteContext) -> RouteDecision | None:
        if len(ctx.history) == 0:
            return RouteDecision(model=self.big)
        return RouteDecision(model=self.small)

router = FirstCallBigRouter(big="gpt-4o", small="gpt-4o-mini")
with router:
    answer = agent.run(question)
```

Return `None` to leave `request_body["model"]` untouched.

Exceptions raised from `route()` are caught and logged — the user's request proceeds unrouted. A router should never break an agent.

---

## How it works

The swap happens in the proxy, right after it parses the inbound request body and before the upstream call. See [How the proxy intercepts calls](proxy.md) for the underlying mechanism.

```
agent.run(q)  →  httpx.Client.send  →  proxy (DIRECT or CONNECT)
                                          ↓
                                  parse body  →  request_body: dict
                                          ↓
                                  router.route(ctx)
                                          ↓
                            request_body["model"] = decision.model
                            body = json.dumps(request_body).encode()
                                          ↓
                                  forward to upstream
                                          ↓
                                  record CallRecord:
                                    model          = routed model
                                    requested_model = original ask
```

Activating a router via `with router:` does three things:

1. **Lazy-starts a singleton proxy** the first time it's used in the process. The proxy stays alive until interpreter exit (`atexit`) — subsequent `with router:` blocks are cheap.
2. **Opens a fresh session** on that proxy, binds the router to it, and sets the httpx redirect ContextVar to the session's port.
3. **Reverses on exit** — closes the session, detaches the router, resets the ContextVar.

Not re-entrant on a single instance. If you need nested or concurrent scopes, instantiate a separate router per scope.

---

## Token accounting

After a swap, `CallRecord` carries both sides:

| Field | Meaning |
|:---|:---|
| `CallRecord.model` | The model actually called upstream. Use this for cost attribution. |
| `CallRecord.requested_model` | The model the client originally asked for, if a router swapped it. `None` means no routing happened or the router declined. |

When running under `LLMTracker`, `get_usage()` groups tokens by `CallRecord.model`, so token and cost totals reflect reality — not the client's original ask.

```python
# If 70 calls were routed to gpt-4o-mini and 30 stayed on gpt-4o:
tracker.get_usage(combo_id="default")
# {"gpt-4o-mini": (14 200, 5 600), "gpt-4o": (6 100, 2 400)}
```

The response cache is keyed on the *routed* model, so a cached `gpt-4o-mini` response is never served when the router chose `gpt-4o`.

---

## Using a router in an experiment (`LLMTracker`)

For ad-hoc agent runs, `with router:` is the whole story. Inside a [model-selection experiment](selectors.md) you want per-call records attributed to the combo you're evaluating — so the router attaches to the tracker instead of being entered standalone:

```python
from agentopt import LLMTracker, RandomRouter

tracker = LLMTracker(router=RandomRouter(model_candidates=["gpt-4o", "gpt-4o-mini"]))
tracker.start()

with tracker.track(data_id="dp_1", combo_id="default"):
    agent.run(input_data)   # routing + attribution, in one session

tracker.stop()
```

`RouteContext` still receives the session's `data_id` / `combo_id` / `agent_id`, so a policy can branch per datapoint or per combo. Don't nest `with router:` inside `tracker.track(...)` — pick one entry point per run.

---

## Scope and limits (v1)

- **Same-provider only.** A router may swap models within one provider (sonnet ↔ haiku, 4o ↔ 4o-mini); it may not swap across providers. Setting `RouteDecision.provider` or `api_key` raises `NotImplementedError`. Those fields exist today so the API can grow later without a break.
- **No reward / learning loop.** `Router` has `route()` only; no `observe(reward)`. Stateful policies manage their own state.
- **HTTP-only.** Routing operates on LLM HTTP requests. Non-HTTP transports (gRPC, WebSocket) are out of scope — same as the rest of the proxy.
