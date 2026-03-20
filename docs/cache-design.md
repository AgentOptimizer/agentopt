# API-Level Response Cache

> **Purpose:** Avoid redundant LLM API calls during model selection evaluation. When the same request (model + messages + parameters) is seen again, the cache returns the stored response instantly — saving cost and wall-clock time without changing observed metrics.

---

## 1. Overview

During model selection, the evaluation harness runs the same dataset across multiple model combinations. Many combinations share identical sub-calls (e.g. a shared planner prompt). Without caching, these redundant calls hit the API, wasting money and time.

The cache sits at the HTTP layer inside the agentopt.proxy interceptor. It is transparent to agent code and framework internals — no code changes required.

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

Two main classes in `src/agentopt/proxy/cache.py`:

### `ResponseCache`

Thread-safe in-memory cache with lazy SQLite persistence.

```python
class ResponseCache:
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        flush_interval: float = 10.0,
    ):
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._dirty: Set[str] = set()
```

- **`get(key)`** — returns `CacheEntry` on hit, `None` on miss. Pure in-memory lookup.
- **`put(key, entry)`** — stores entry in memory, marks key as dirty for next flush.
- **`flush()`** — writes all dirty entries to SQLite in a single transaction.
- **`clear()`** — clears all entries from memory and the database.
- **`close()`** — flushes pending writes and stops the background thread.
- All in-memory operations are protected by `threading.Lock`.

### `CacheEntry`

```python
@dataclass
class CacheEntry:
    response_bytes: bytes           # raw HTTP response body
    response_headers: Dict[str, str]  # HTTP headers (minus encoding headers)
    latency_seconds: float          # original API call latency
```

The original latency is stored so it can be replayed on cache hits for fair metrics.

---

## 3. Persistence: SQLite Backend

When `cache_dir` is provided, the cache persists entries to a SQLite database (`cache.db`) inside that directory.

### Why SQLite over individual files?

A typical evaluation run produces 10K–50K responses. Individual JSON files cause:
- Slow `glob()` / `ls` at 50K+ files in a flat directory
- Slow startup (sequential file reads)
- Filesystem overhead on some platforms (e.g. older macOS HFS+)

SQLite provides:
- **Single file** on disk — clean, portable
- **Fast indexed lookups** by primary key
- **Atomic batch writes** via transactions
- **No extra dependencies** — `sqlite3` ships with Python's standard library
- **WAL journal mode** for better concurrent read performance

### Schema

```sql
CREATE TABLE IF NOT EXISTS cache (
    key       TEXT PRIMARY KEY,
    data_json TEXT NOT NULL
)
```

Each row stores a cache key (SHA-256 hash) and the JSON-serialized `CacheEntry`.

### Flush strategy

Writes are **lazy** — `put()` only updates the in-memory dict and marks the key as dirty. Dirty entries are flushed to SQLite:

- **Periodically** by a background daemon thread (every 10 seconds by default)
- **Explicitly** via `flush()`
- **Automatically** when `close()` is called (which `LLMTracker.stop()` calls)

This keeps `put()` and `get()` free of disk I/O on the hot path.

### Default behavior

`LLMTracker` defaults to `cache_dir=".agentopt_cache"`, so the cache persists across process restarts automatically. Set `cache_dir=None` for in-memory only.

```python
tracker = LLMTracker()                          # cache on, persists to .agentopt_cache/
tracker = LLMTracker(cache_dir="./my_cache")    # custom directory
tracker = LLMTracker(cache_dir=None)            # in-memory only
tracker = LLMTracker(cache=False)               # cache off entirely
```

---

## 4. Cache Key Design

```python
def _make_cache_key(request_body: dict) -> str:
    body = {k: v for k, v in request_body.items() if k != "stream"}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()
```

- **Deterministic:** `json.dumps` with `sort_keys=True` ensures identical requests produce identical keys regardless of dict ordering.
- **Excludes `stream`:** The `stream` field is ephemeral metadata that doesn't affect the response content.
- **SHA256:** Strong hash with negligible collision probability.

**Included in the key:** model, messages, temperature, top_p, max_tokens, and all other request body fields.

---

## 5. Cache Policy

| Condition | Cached? | Reason |
|-----------|---------|--------|
| Non-streaming, HTTP 200 | Yes | Complete response available |
| Streaming request | No | Response body is incomplete at `send()` return time |
| Non-200 status | No | Error responses should not be replayed |

---

## 6. Latency Replay

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

## 7. Integration with Interceptor

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

## 8. LLMTracker API

### Construction

```python
tracker = LLMTracker()                          # cache on, disk persistence to .agentopt_cache/
tracker = LLMTracker(cache_dir="./my_cache")    # custom cache directory
tracker = LLMTracker(cache_dir=None)            # in-memory only
tracker = LLMTracker(cache=False)               # cache off
```

### Runtime control

```python
tracker.flush_cache()           # flush dirty entries to disk immediately
tracker.clear_cache()           # clear all entries and delete from DB
```

### Latency accounting

```python
cached_latency = tracker.get_cached_latency(data_id="dp_1", combo_id="gpt4o+haiku")
# Sum of latency_seconds for all cached=True records matching the filters
```

---

## 9. Thread Safety

The cache is designed for concurrent access during parallel model selection:

- **`ResponseCache`** uses `threading.Lock()` on all in-memory operations (`get`, `put`, `clear`, `__len__`).
- **SQLite writes** happen only during `flush()`, using a fresh connection per flush. SQLite's WAL mode allows concurrent reads.
- **Global cache instance** in the interceptor module is set during `install()` and read during active patching. It is not modified during patching.
- **ContextVars** provide per-thread/per-task isolation for attribution, so parallel evaluations with different `combo_id` values never interfere with each other's cache lookups.

---

## 10. Limitations

- **No TTL / expiry** — entries persist until explicitly cleared. Appropriate for eval loops that run with fresh data.
- **No streaming** — streaming responses are incomplete at `send()` return time and cannot be cached.
- **No selective caching** — all non-streaming, status-200 requests are cached when caching is enabled. No per-model or per-endpoint filtering.
