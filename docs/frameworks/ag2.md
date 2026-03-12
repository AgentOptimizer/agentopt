# AG2 (AutoGen 2)

AG2 validates `llm_config` and rejects `ModelProxy`, so AgentOpt patches the validation at import time. Pass an `LLMConfig` to `ModelProxy`:

## Installation

```bash
uv sync --extra ag2
```

## Example

```python
from autogen import ConversableAgent, LLMConfig
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLMConfig
proxy = ModelProxy(LLMConfig({"model": "gpt-4o-mini", "api_key": "..."}))

# 2. Build agent (patched validation accepts ModelProxy)
agent = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=proxy,
    human_input_mode="NEVER",
)

# 3. Run optimization
selector = ModelSelector(
    models={proxy: ["gpt-4o-mini", "anthropic/claude-sonnet-4-20250514"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=agent,
)
results = selector.select_best()
```

## Limitations

- Supports OpenAI and Anthropic models natively
- Other providers not yet supported
