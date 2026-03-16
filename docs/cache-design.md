# API-Level Response Cache

> **Purpose:** Avoid redundant LLM API calls during model selection evaluation. When the same request (model + messages + parameters) is seen again, the cache returns the stored response instantly — saving cost and wall-clock time without changing observed metrics.

---

## 1. Overview

During model selection, the evaluation harness runs the same dataset across multiple model combinations. Many combinations share identical sub-calls (e.g. a shared planner prompt). Without caching, these redundant calls hit the API, wasting money and time.

The cache sits at the HTTP layer inside agentproxy's interceptor. It is transparent to agent code and framework internals — no code changes required.

```
agent(input_data)
  └── framework internals
        └── httpx.Client.send()
              ├── cache hit?  → return stored response
              └── cache miss? → real API call → store response
```

Caching is enabled by default via `LLMTracker(cache=True)`.

---

## 2. Architecture

Three classes in `agentproxy/src/agentproxy/cache.py`:

### `ResponseCache`

Thread-safe in-memory LRU cache built on `collections.OrderedDict`.

```python
class ResponseCache:
    def __init__(self, max_size: int = 0):  # 0 = unlimited
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self.stats = CacheStats()
```

- **`get(key)`** — returns `CacheEntry` on hit (moves to end for LRU), increments `stats.hits`. Returns `None` on miss, increments `stats.misses`.
- **`put(key, entry)`** — stores entry, evicts oldest if over `max_size`.
- **`clear()`** — clears all entries and resets stats.
- All operations are protected by `threading.Lock`.

### `CacheEntry`

```python
@dataclass
class CacheEntry:
    response_bytes: bytes           # raw HTTP response body
    response_headers: Dict[str, str]  # HTTP headers (minus encoding headers)
    latency_seconds: float          # original API call latency
```

The original latency is stored so it can be replayed on cache hits for fair metrics.

### `CacheStats`

```python
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int: ...      # hits + misses
    @property
    def hit_rate(self) -> float: ... # hits / total (0.0 if no lookups)
```

---

## 3. Cache Key Design

```python
def _make_cache_key(request_body: dict) -> str:
    body = {k: v for k, v in request_body.items() if k != "stream"}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()
```

- **Deterministic:** `json.dumps` with `sort_keys=True` ensures identical requests produce identical keys regardless of dict ordering.
- **Excludes `stream`:** The `stream` field is ephemeral metadata that doesn't affect the response content. A non-streaming request should match a previously cached non-streaming request with the same model/messages/parameters.
- **SHA256:** Strong hash with negligible collision probability.

**Included in the key:** model, messages, temperature, top_p, max_tokens, and all other request body fields.

---

## 4. Cache Policy

| Condition | Cached? | Reason |
|-----------|---------|--------|
| Non-streaming, HTTP 200 | Yes | Complete response available |
| Streaming request | No | Response body is incomplete at `send()` return time |
| Non-200 status | No | Error responses should not be replayed |

**Max size:** Configurable via `cache_max_size` constructor parameter. `0` (default) means unlimited. When a positive limit is set, the oldest entry (LRU) is evicted on overflow.

---

## 5. Latency Replay

A key design decision: **cached responses replay the original API call latency** in the `CallRecord`.

When a cache hit occurs:
1. The `CacheEntry.latency_seconds` (from the original API call) is recorded in the `CallRecord`
2. The `CallRecord.cached` flag is set to `True`
3. The actual wall-clock time for the cache hit is near-zero

In the model selection evaluation loop (`BaseModelSelector._evaluate_agent()`):

```python
wall_clock = time.time() - start_time
cached_latency = self._tracker.get_cached_latency(data_id=dp_id)
latency = wall_clock + cached_latency  # add back cache savings
```

This ensures that latency metrics reflect what a **real, uncached run** would cost. Without this correction, cached combinations would appear artificially fast, making comparisons unfair.

---

## 6. Integration with Interceptor

The cache integrates into the patched `httpx.Client.send()` via two helper functions in `interceptor.py`:

### Request flow

```
_patched_send(request)
  │
  ├── Is this an LLM endpoint? (/chat/completions or /v1/messages)
  │     No → call original send(), return
  │
  ├── Is streaming? (stream=True in body)
  │     Yes → call original send(), record metrics, return
  │
  ├── _try_cache_lookup(request_body)
  │     Hit → construct synthetic httpx.Response from cached bytes
  │          → _try_record() with cached=True
  │          → return cached response
  │
  ├── Call original send() [real API call, measure latency]
  │
  ├── Status 200?
  │     Yes → _try_cache_store(request_body, response, latency)
  │          → _try_record() with cached=False
  │
  └── return response
```

### `_try_cache_lookup()`

- Computes cache key from request body
- Returns `(httpx.Response, original_latency)` on hit, `None` on miss
- Constructs a synthetic `httpx.Response` from stored bytes
- **Strips `content-encoding` and `transfer-encoding` headers** to prevent double-decompression

### `_try_cache_store()`

- Stores response bytes, headers, and latency as a `CacheEntry`
- Only called for HTTP 200, non-streaming responses

Both sync (`httpx.Client.send`) and async (`httpx.AsyncClient.send`) paths use the same cache instance.

---

## 7. LLMTracker API

### Construction

```python
tracker = LLMTracker()                          # cache on, unlimited size
tracker = LLMTracker(cache=True, cache_max_size=1000)  # cache on, max 1000 entries
tracker = LLMTracker(cache=False)               # cache off
```

### Runtime control

```python
tracker.cache_enabled = False   # disable without clearing
tracker.cache_enabled = True    # re-enable

tracker.clear_cache()           # clear all entries and reset stats
```

**Constraint:** Cannot enable caching if the tracker was constructed with `cache=False`. Attempting to do so raises `RuntimeError` because no `ResponseCache` was initialized.

### Statistics

```python
stats = tracker.cache_stats   # CacheStats object
stats.hits                    # number of cache hits
stats.misses                  # number of cache misses
stats.total                   # hits + misses
stats.hit_rate                # hits / total (0.0 if no lookups)
```

### Latency accounting

```python
cached_latency = tracker.get_cached_latency(data_id="dp_1", combo_id="gpt4o+haiku")
# Sum of latency_seconds for all cached=True records matching the filters
```

---

## 8. Thread Safety

The cache is designed for concurrent access during parallel model selection:

- **`ResponseCache`** uses `threading.Lock()` on all operations (`get`, `put`, `clear`, `__len__`). LRU updates (`move_to_end`) happen within the lock.
- **Global cache instance** in the interceptor module is set during `install()` and read during active patching. It is not modified during patching.
- **ContextVars** provide per-thread/per-task isolation for attribution, so parallel evaluations with different `combo_id` values never interfere with each other's cache lookups.

---

## 9. Limitations

- **In-memory only** — cache is lost on process restart. Suitable for evaluation loops where data fits in RAM.
- **No TTL / expiry** — entries persist until evicted by LRU or explicitly cleared. Appropriate for eval loops that run with fresh data.
- **No streaming** — streaming responses are incomplete at `send()` return time and cannot be cached.
- **No selective caching** — all non-streaming, status-200 requests are cached when caching is enabled. No per-model or per-endpoint filtering.
