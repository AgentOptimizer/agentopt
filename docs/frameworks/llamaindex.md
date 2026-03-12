# LlamaIndex

LlamaIndex uses strict Pydantic v2 validation, so `ModelProxy` cannot be passed directly as the `llm` parameter. Instead, wrap the LLM *after* agent creation.

## Installation

```bash
uv sync --extra llamaindex
```

## Example

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from agentopt import ModelProxy, ModelSelector
from agentopt.model_proxy.framework_specific_implementation.llamaindex import build_llamaindex_llm

# 1. Create a real LLM first
real_llm = build_llamaindex_llm("gpt-4o-mini")

# 2. Build your agent normally
agent = FunctionAgent(
    tools=my_tools,
    llm=real_llm,
    system_prompt="You are a helpful assistant.",
)
workflow = AgentWorkflow(agents=[agent], root_agent=agent.name)

# 3. Wrap the LLM after agent creation
llm = ModelProxy(real_llm)

# 4. Run optimization
selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=workflow,
)
results = selector.select_best()
```

## Why wrap after creation?

LlamaIndex's `FunctionAgent` validates the `llm` parameter type at construction using Pydantic. Since `ModelProxy` is not a recognized LLM type, passing it directly raises a validation error. By wrapping after creation, the proxy replaces the reference without triggering validation.

## How it works

- **Detection:** `LlamaIndexAdapter.detect()` checks for LlamaIndex workflow classes
- **Invocation:** Calls `workflow.run()` (async, wrapped automatically)
- **Model building:** `build_llamaindex_llm()` creates the correct LlamaIndex LLM class with OpenRouter fallback
