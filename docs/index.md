# AgentOpt

**Find the right LLM models for your AI agents.**

Choosing the right LLM model is hard. Different models have different cost, performance, and latency tradeoffs. Should you use a thinking model? What effort level? What about different models for different steps of your agent pipeline? The combinatorial space explodes quickly — if your agent has 3 steps and you're considering 5 models per step, that's 125 combinations to evaluate.

AgentOpt solves this automatically. Give it your agent and a small evaluation dataset (~100 samples), and it will efficiently search the model combination space to present you with the **Pareto curve of accuracy/cost/latency tradeoffs** — so you can make an informed choice.

## Key Features

- **Non-intrusive**: Wrap your agent in a simple factory function — we take care of the rest. No framework adapters, no code changes to your agent internals.
- **Framework-agnostic**: Works with OpenAI SDK, LangChain, LangGraph, CrewAI, LlamaIndex, AG2, or any framework that uses `httpx` for LLM calls.
- **Smart search algorithms**: 7 selection algorithms from brute force to Bayesian optimization, so you don't have to evaluate every combination.
- **Automatic tracking**: Transparently intercepts all LLM calls to measure token usage, latency, and cost — no manual instrumentation.
- **Response caching**: Identical LLM calls are cached (in-memory + SQLite on disk), so re-running experiments is instant and free.

## Quick Example

```python
from agentopt import BruteForceModelSelector

selector = BruteForceModelSelector(
    agent_fn=agent_maker,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        "solver":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)

results = selector.select_best(parallel=True)
results.print_summary()
```

```
    Model Selection Results
    ----------------------------------------------------------------------------
    Rank  Model                                     Accuracy  Latency      Price
    ----------------------------------------------------------------------------
>>>    1  planner=gpt-4.1-nano + solver=gpt-4.1-nano 100.00%    0.85s  $0.000420
       2  planner=gpt-4o-mini + solver=gpt-4o-mini   100.00%    1.20s  $0.002372
       3  planner=gpt-4o + solver=gpt-4o              100.00%    2.70s  $0.014355
    ...
```

## Next Steps

- [Installation](getting-started/installation.md) — Install AgentOpt
- [Quick Start](getting-started/quickstart.md) — Build and optimize your first agent
- [Selection Algorithms](concepts/algorithms.md) — Choose the right search strategy
