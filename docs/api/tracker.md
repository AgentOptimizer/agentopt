# Tracker

`LLMTracker` is the single primitive that captures LLM calls — by token usage, latency, model, and request/response — across every agent shape AgentOpt supports (in-process SDKs, subprocess CLIs, container agents). Selection and routing both build on it; you can also use it standalone for tracking and caching.

```python
from agentopt import LLMTracker
```

The same code runs against the in-process proxy (default) or a long-lived gateway daemon (`AGENTOPT_GATEWAY_URL=…`). The env var is the entire deployment switch — no API change.

---

## Quick patterns

### Plain tracking (no router, no selector)

```python
with LLMTracker(combo_id="run-1") as tracker:
    agent.run(prompt)
print(tracker.get_usage())
```

### Per-datapoint sessions

```python
with LLMTracker() as tracker:
    for i, q in enumerate(questions, 1):
        with tracker.track(data_id=f"q{i}"):
            agent.run(q)
tracker.print_summary()         # grouped by data_id
```

### With a router

```python
from agentopt import LLMTracker, RandomRouter

router = RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"], seed=0)
with LLMTracker(router=router) as tracker:
    for i, q in enumerate(questions, 1):
        with tracker.track(data_id=f"q{i}"):
            agent.run(q)
tracker.print_summary()
```

`router=` alone is **not** single-session sugar — it sets a default that nested `tracker.track()` calls inherit, so the host pattern above works.

### Subprocess agent

```python
import agentopt, subprocess

with LLMTracker(combo_id="subproc") as tracker:
    proxy = agentopt.get_current_session_proxy()
    env = {**os.environ, **proxy.env_dict()}
    subprocess.run(["claude", "-p", prompt], env=env)
```

The session's mitmproxy port is exposed via `HTTPS_PROXY`; the CA bundle path via `SSL_CERT_FILE` (and friends for non-stdlib HTTP clients). See [proxy.md](proxy.md) for the full subprocess flow.

---

## Constructor

```python
LLMTracker(
    *,
    data_id: str | None = None,
    combo_id: str | None = None,
    agent_id: str | None = None,
    router: Router | None = None,
    cache: bool = True,
    cache_dir: str | Path | None = ".agentopt_cache",
)
```

| Param | Description |
|:---|:---|
| `data_id` / `combo_id` / `agent_id` | If **any** is set, `__enter__` auto-opens a single tracking session with those IDs and `__exit__` closes it. The "single-session sugar" path. |
| `router` | A [`Router`](router.md) to apply at the proxy layer. On its own it sets a default that nested `track()` calls inherit; combined with one of the IDs above it also routes the auto-opened session. |
| `cache` | Enable response caching. Hits short-circuit before any network round-trip. |
| `cache_dir` | Persist cache to disk at this path. Pass `None` to keep it in memory only. |

`AGENTOPT_GATEWAY_URL` is read in `__init__`; when set, the tracker delegates to `RemoteBackend` and `cache_dir` is honored by the daemon, not the client.

---

## Lifecycle

| Method | Description |
|:---|:---|
| `start()` | Install the httpx redirect and prepare the backend. Idempotent. |
| `stop()` | Tear down live sessions, restore httpx, flush cache. **Record queries remain valid after `stop()`** — so `tracker.print_summary()` works right after a `with` block exits. |
| `close()` | Final teardown. For `RemoteBackend` this drops the long-lived control-plane HTTP client; for `LocalBackend` it's equivalent to `stop()`. Idempotent. |

`__exit__` calls `stop()`, **not** `close()` — so post-`with` queries against the daemon don't fail with "client closed". Call `tracker.close()` explicitly when you want to release the HTTP client (or let `__del__` do it).

---

## Sessions

```python
@contextmanager
tracker.track(
    data_id: str | None = None,
    combo_id: str | None = None,
    agent_id: str | None = None,
    router: Router | None = None,
) -> SessionInfo
```

Open a tracking session. In local mode this eagerly spins up a per-session mitmproxy `SessionMaster` on an ephemeral port and sets a `ContextVar` so in-process httpx calls are attributed correctly. In daemon mode it POSTs `/sessions` and reuses the daemon's port.

All four params are optional. `router=` falls back to the `router` passed to `LLMTracker(...)` if you don't supply one here.

| Helper | Description |
|:---|:---|
| `get_session_env(session)` | `{HTTPS_PROXY, SSL_CERT_FILE, REQUESTS_CA_BUNDLE, NODE_EXTRA_CA_CERTS}` for subprocess agents. Identical shape in both modes. |
| `agentopt.get_current_session_proxy()` | Module-level convenience — reads the active session from the `ContextVar` and returns a `SessionProxy` (`url`, `port`, `ca_pem`, `ca_bundle_path`, `env_dict()`). Returns `None` outside a `track()` scope. |

---

## Queries

| Method | Returns | Description |
|:---|:---|:---|
| `records` (property) | `List[CallRecord]` | All records captured so far. |
| `get_records(data_id=None, combo_id=None, agent_id=None)` | `List[CallRecord]` | Filtered records. |
| `get_usage(...)` | `Dict[str, Tuple[int, int]]` | `{model: (input_tokens, output_tokens)}` aggregated over matching records. |
| `get_cached_latency(...)` | `float` | Total latency (seconds) of cache-hit records — useful for "how much wall time did caching save?" |
| `print_summary(data_id=None, combo_id=None, agent_id=None)` | `None` | Model sequence, per-model tokens, and total latency. Grouped by `data_id` when records span multiple distinct values; flat otherwise. Thin wrapper over `agentopt.routing.print_routing_summary`. |

---

## Cache & providers

| Method | Description |
|:---|:---|
| `flush_cache()` | Flush dirty cache entries to disk. |
| `clear_cache()` | Drop all cached responses (memory + disk). |
| `clear()` | Clear locally archived records. |
| `register_provider(name, base_url, path_patterns)` | Add or replace an LLM provider so its hostnames are MITM-intercepted and its paths recognized by the in-process patch. In daemon mode this also POSTs `/providers` to keep the daemon in sync. |

Cache keys hash the routed model (not the requested one), so a router swapping `gpt-4o → gpt-4o-mini` produces a distinct cache entry. See [router.md](router.md#how-it-works).

---

## CallRecord

```python
from agentopt import CallRecord
```

| Field | Type | Description |
|:---|:---|:---|
| `data_id` | `str?` | Datapoint identifier. |
| `combo_id` | `str?` | Model-combination identifier. |
| `agent_id` | `str?` | Agent role identifier. |
| `model` | `str` | Model name (the one actually used after any routing). |
| `prompt_tokens` | `int` | Input tokens. |
| `completion_tokens` | `int` | Output tokens. |
| `latency_seconds` | `float` | API call duration. |
| `request_url` | `str` | Upstream URL. |
| `request_body` | `dict` | Parsed request payload. |
| `response_body` | `dict` | Parsed response payload. |
| `timestamp` | `str` | ISO 8601. |
| `cached` | `bool` | Whether this was a cache hit. |
| `error` | `str?` | Set when the upstream failed or token extraction couldn't parse a successful response. The model name is `"<parse-failed>"` in the latter case so the failure surfaces in summaries. |

---

## Use with `ModelSelector`

`ModelSelector` instances accept a `tracker=` kwarg. By default they construct one internally and call `start()` in the constructor / `stop()` when `select_best()` returns. Pass your own when you want to share a cache across runs, point at a daemon, or post-process the records after selection completes:

```python
tracker = LLMTracker(cache_dir="./shared_cache")
selector = ModelSelector(agent=..., models=..., tracker=tracker, ...)
results = selector.select_best()
# tracker has been stopped, but records remain queryable:
print(tracker.get_usage())
```

See [selectors.md](selectors.md) for the full constructor surface.

---

## ResponseCache (low-level)

```python
from agentopt.proxy import ResponseCache
```

Usually owned by `LLMTracker`. Exposed for tests and advanced setups:

| Method | Description |
|:---|:---|
| `get(key)` | Look up a cached entry. |
| `put(key, entry)` | Store an entry (dirty until `flush`). |
| `flush()` | Write dirty entries to SQLite. |
| `clear()` | Drop memory + disk. |
| `close()` | Flush and stop the background flush thread. |
