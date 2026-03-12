# Claude Agent SDK

The Claude SDK uses a functional API (no persistent agent object), so use `invoke_fn`:

## Installation

```bash
uv sync --extra claude-agent-sdk
```

## Example

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, query
from agentopt import ModelProxy, ModelSelector

proxy = ModelProxy(ClaudeAgentOptions(model="haiku"))

async def _query_async(prompt, options):
    result = ""
    async for msg in query(prompt=prompt, options=options):
        if hasattr(msg, "result"):
            result = msg.result
    return result

def invoke_fn(input_data):
    return asyncio.run(_query_async(input_data["input"], proxy))

selector = ModelSelector(
    models={proxy: ["haiku", "sonnet"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    invoke_fn=invoke_fn,
)
results = selector.select_best(parallel=True)
```

## Limitations

- Only supports Claude models
- Uses short aliases: `"haiku"`, `"sonnet"`, `"opus"`
