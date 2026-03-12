# OpenAI Agents SDK

AgentOpt registers `ModelProxy` as a virtual subclass of the OpenAI SDK's `Model` ABC, so it can be passed directly as the `model` parameter.

## Installation

```bash
uv sync --extra openai-agents
```

## Example

```python
from types import SimpleNamespace
from agents import Agent
from agentopt import ModelProxy, ModelSelector

# 1. Wrap a model spec
proxy = ModelProxy(SimpleNamespace(model="gpt-4o-mini"))

# 2. Build your agent
agent = Agent(name="Math QA", model=proxy, instructions="Answer math questions concisely.")

# 3. Run optimization
selector = ModelSelector(
    models={proxy: ["gpt-4o-mini", "gpt-4o"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=agent,
)
results = selector.select_best()
```

## Limitations

- Only supports OpenAI models (the SDK routes through OpenAI's API)
- Cross-provider model names (e.g., `anthropic/...`) are not supported
