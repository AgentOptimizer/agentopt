# AgentOpt

**Framework-agnostic optimization for LLM-powered agents.**

AgentOpt evaluates and selects the best models for your AI agents across frameworks like CrewAI, LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, and AG2 — or any custom agent setup.

!!! note
    This project is in early development. APIs are subject to change.

## How it works

AgentOpt works in three steps:

1. **Wrap** your LLM with [`ModelProxy`](concepts/model-proxy.md) — a transparent proxy that forwards all calls
2. **Build** your agent as usual — the proxy is invisible to your framework
3. **Run** a [`ModelSelector`](concepts/model-selection.md) to find the best model by accuracy, latency, and cost

```python
from agentopt import ModelProxy, ModelSelector

# Wrap any LLM
llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# Build your agent normally — proxy is transparent
agent = build_my_agent(llm)

# Find the best model
selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]},
    eval_fn=lambda expected, actual: expected.lower() in actual.lower(),
    dataset=my_dataset,
    agent=agent,
)
results = selector.select_best()
results.print_summary()
```

## Key features

- **Framework-agnostic** — works with CrewAI, LangChain, LangGraph, LlamaIndex, OpenAI SDK, Claude SDK, AG2, or custom agents
- **Multiple selection strategies** — brute force, random search, hill climbing, arm elimination, Hyperband, Bayesian optimization
- **Parallel evaluation** — evaluate model combinations concurrently with thread-safe model overrides
- **Multi-agent optimization** — optimize multiple LLMs in a pipeline simultaneously (Cartesian product search)
- **Response caching** — cache LLM responses to avoid redundant API calls during benchmarking
- **Token tracking** — monitor input/output token usage per model

## Quick links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Framework Guides](frameworks/overview.md)
- [API Reference](api/index.md)
