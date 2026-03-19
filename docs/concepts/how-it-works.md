# How It Works

## The Interception Layer

Every major LLM SDK — OpenAI, Anthropic, LangChain, CrewAI, LlamaIndex — uses Python's `httpx` library for HTTP requests. AgentOpt patches `httpx.Client.send()` and `httpx.AsyncClient.send()` at the class level, intercepting every LLM API call in the process:

```
your_agent(input)
  └── framework internals (LangChain, CrewAI, etc.)
        └── httpx.Client.send()   ← intercepted here
              └── LLM API (OpenAI, Anthropic, etc.)
```

This design means:

- **Zero code changes** to your agent or framework
- **No proxy server** to run or configure
- **Works with any framework** that uses httpx (which is all of them)

## What Gets Tracked

For each intercepted LLM call, AgentOpt records:

| Field | Description |
|-------|-------------|
| `model` | Model name (e.g., `gpt-4o`) |
| `prompt_tokens` | Input token count |
| `completion_tokens` | Output token count |
| `latency_seconds` | Wall-clock time for the API call |
| `data_id` | Which datapoint triggered this call |
| `combo_id` | Which model combination was being evaluated |
| `cached` | Whether this was a cache hit |

## Attribution via ContextVars

AgentOpt uses Python's `contextvars` to attribute LLM calls to the right datapoint and model combination. Each thread and async task gets its own independent context, so parallel evaluations never interfere with each other.

```python
with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku"):
    result = agent(input_data)
    # All LLM calls made inside this block are tagged with dp_1 + gpt4o+haiku
```

## Response Caching

AgentOpt caches LLM responses at the API level:

- **Cache key**: Hash of the request body (model + messages + parameters), excluding the `stream` field
- **In-memory**: All entries kept in a thread-safe dict
- **On disk** (optional): SQLite database, flushed periodically by a background thread

When the same prompt is sent to the same model, the cached response is returned instantly — preserving the original latency measurement so cost/latency comparisons remain valid.

This means:

- Re-running an experiment after a crash costs nothing
- Iterating on your evaluation function doesn't re-call the API
- Shared model calls across combinations are free (e.g., if two combinations use the same planner model with the same input)

## The Selection Loop

For each model combination, AgentOpt:

1. Calls your `agent_fn(models)` to build an agent with the candidate models
2. Runs the agent on every datapoint in your evaluation dataset
3. Tracks token usage, latency, and cost via the interception layer
4. Scores each output using your `eval_fn(expected, actual)`
5. Aggregates accuracy, average latency, and total cost per combination
6. Reports results ranked by accuracy (ties broken by latency)

Different [selection algorithms](algorithms.md) vary in how they choose which combinations to evaluate, but the evaluation loop is the same.
