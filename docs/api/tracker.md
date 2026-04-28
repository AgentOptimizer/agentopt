# Tracker

The `LLMTracker` class handles LLM call interception, recording, and caching. It lives in `agentopt.proxy`.

## LLMTracker

```python
from agentopt.proxy import LLMTracker
```

### Constructor

```python
tracker = LLMTracker(
    cache=True,                      # Enable response caching (default: True)
    cache_dir=".agentopt_cache",     # Persist cache to disk (default: ".agentopt_cache")
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

### Example: Gemini CLI (subprocess)

```python
import os
import subprocess

from agentopt.proxy import LLMTracker

tracker = LLMTracker(cache=False)
tracker.start()

try:
    with tracker.track(data_id="dp_1", combo_id="gemini-cli") as session:
        env = {**os.environ, **tracker.get_session_env(session)}
        subprocess.run(
            ["gemini", "prompt", "Write one sentence about model selection."],
            env=env,
            check=True,
        )

    records = tracker.get_records(data_id="dp_1", combo_id="gemini-cli")
    print("calls:", len(records))
    for r in records:
        print(r.model, r.prompt_tokens, r.completion_tokens, f"{r.latency_seconds:.2f}s")
finally:
    tracker.stop()
```

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
