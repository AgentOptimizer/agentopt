# Response Caching

AgentOpt caches LLM responses at the API level to avoid redundant calls during model selection.

## How It Works

- **Cache key**: SHA-256 hash of the request body (model + messages + parameters), with the `stream` field excluded
- **In-memory**: All entries stored in a thread-safe dict for fast lookup
- **On disk** (optional): SQLite database (`cache.db`), flushed periodically by a background thread

When the same prompt is sent to the same model with the same parameters, the cached response is returned instantly. The original latency is preserved in the cached entry so that cost/latency comparisons remain valid across combinations.

## Why Caching Matters

During model selection, many LLM calls are repeated:

- **Shared prefixes**: If two model combinations use the same planner model, the planner call for a given datapoint is identical
- **Re-runs**: If you tweak your eval function and re-run, all LLM calls are cache hits
- **Crash recovery**: If a run is interrupted, cached responses survive on disk

## Enabling Disk Cache

By default, caching is in-memory only (lost when the process exits). To persist to disk:

```python
from agentopt.proxy import LLMTracker

tracker = LLMTracker(cache_dir="./llm_cache")
selector = BruteForceModelSelector(
    ...,
    tracker=tracker,
)
results = selector.select_best()
# Cache automatically flushed to ./llm_cache/cache.db
```

The cache is stored as a single SQLite database file. On subsequent runs with the same `cache_dir`, entries are loaded from disk on startup.

## Cache Lifecycle

| Event | What happens |
|-------|-------------|
| `LLMTracker(cache_dir=...)` | Creates DB, loads existing entries into memory |
| LLM call (cache miss) | Response stored in memory, marked dirty |
| Background flush (every 10s) | Dirty entries written to SQLite |
| `tracker.stop()` / `select_best()` returns | Final flush to disk |
| `tracker.clear_cache()` | Clears memory and deletes all DB rows |

## Disabling Cache

```python
tracker = LLMTracker(cache=False)
```

## Inspecting the Cache

The cache is a standard SQLite database:

```bash
sqlite3 ./llm_cache/cache.db "SELECT COUNT(*) FROM cache"
```
