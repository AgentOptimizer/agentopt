# Response Caching

AgentOpt includes a response cache to avoid redundant API calls when re-running benchmarks or comparing selection strategies.

## Overview

The cache stores LLM responses keyed by `(model_name, prompt)` pairs. Each entry also records the original call latency, so that benchmark timing remains accurate on cached runs.

## Usage

```python
from agentopt import ResponseCache, NoCache

# Create a file-backed cache
cache = ResponseCache("path/to/cache.json")

# Generate a deterministic key
key = cache.make_key("gpt-4o", question_text)

# Check for cached response
hit = cache.get(key)
if hit is not None:
    response, latency = hit
    time.sleep(latency)  # preserve original timing
    return response

# On cache miss, call the API and store the result
start = time.time()
response = call_llm(prompt)
latency = time.time() - start

cache.set(key, response, latency, metadata={"model": "gpt-4o"})
```

## Why store latency?

When comparing models on a Pareto curve (accuracy vs. latency), cached runs need to reflect the original API call timing. Without stored latency, a cached run would show near-zero latency for all models, making the comparison meaningless.

By calling `time.sleep(latency)` on cache hits, the outer timing code captures realistic durations.

## NoCache

`NoCache` is a drop-in replacement that never caches. All methods match `ResponseCache` signatures so you can use it without `if` guards:

```python
cache = NoCache() if args.no_cache else ResponseCache("cache.json")

# Same code works with either
key = cache.make_key(model, prompt)
hit = cache.get(key)
```

## Thread safety

`ResponseCache` is thread-safe — all reads and writes are protected by a lock. It can be safely used in parallel evaluation modes.

## Cache key format

Keys are 16-character hex strings — the first 16 characters of `SHA-256("model_name::prompt")`. This is deterministic: the same model + prompt always produces the same key.

## Persistence

The cache is backed by a JSON file. It:

- Writes to disk after every `set()` call
- Creates parent directories automatically
- Recovers gracefully from corrupt JSON files (starts fresh with a warning)

## API

::: agentopt.cache.ResponseCache

::: agentopt.cache.NoCache
