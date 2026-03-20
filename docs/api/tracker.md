# Tracker

The `LLMTracker` class handles LLM call interception, recording, and caching. It lives in `agentopt.proxy`.

## LLMTracker

```python
from agentopt.proxy import LLMTracker
```

### Constructor

```python
tracker = LLMTracker(
    cache=True,              # Enable response caching (default: True)
    cache_dir="./llm_cache", # Persist cache to disk (default: None, memory-only)
)
```

### Methods

| Method | Description |
|:-------|:------------|
| `start()` | Install httpx patches (idempotent) |
| `stop()` | Restore original httpx, flush cache to disk |
| `track(data_id, combo_id, agent_id=None)` | Context manager — attributes all LLM calls in scope |
| `track_agent(agent_id)` | Context manager — sets only agent_id |
| `get_records(data_id=None, combo_id=None)` | Filtered list of `CallRecord` |
| `get_usage(data_id=None, combo_id=None)` | `{model: (input_tokens, output_tokens)}` |
| `flush_cache()` | Write dirty cache entries to disk |
| `clear_cache()` | Clear all cached responses |
| `clear()` | Clear all recorded data |

!!! note "Automatic lifecycle"
    When using a selector, the tracker is managed automatically — `start()` is called in the constructor and `stop()` is called when `select_best()` returns.

---

## CallRecord

```python
from agentopt.proxy import CallRecord
```

| Field | Type | Description |
|:------|:-----|:------------|
| `data_id` | `str?` | Datapoint identifier |
| `combo_id` | `str?` | Model combination identifier |
| `agent_id` | `str?` | Agent role identifier |
| `model` | `str` | Model name |
| `prompt_tokens` | `int` | Input token count |
| `completion_tokens` | `int` | Output token count |
| `latency_seconds` | `float` | API call duration |
| `request_url` | `str` | API endpoint URL |
| `request_body` | `dict` | Full request payload |
| `response_body` | `dict` | Full response payload |
| `timestamp` | `str` | ISO 8601 timestamp |
| `cached` | `bool` | Whether this was a cache hit |

---

## ResponseCache

```python
from agentopt.proxy import ResponseCache
```

Low-level cache API (usually managed by `LLMTracker`):

| Method | Description |
|:-------|:------------|
| `get(key)` | Look up a cached entry |
| `put(key, entry)` | Store an entry (dirty, not yet on disk) |
| `flush()` | Write dirty entries to SQLite |
| `clear()` | Clear memory and disk |
| `close()` | Flush and stop background thread |
