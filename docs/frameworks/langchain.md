# LangChain

AgentOpt integrates with LangChain via duck typing — `ModelProxy` wraps `ChatOpenAI` (or any LangChain LLM) transparently.

## Example

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLM
llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# 2. Build your agent normally
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# 3. Run optimization
selector = ModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]},
    eval_fn=my_eval_fn,
    dataset=dataset,
    agent=executor,
)
results = selector.select_best()
```

## Multi-LLM

```python
planner_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
executor_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

# Build pipeline using both proxies
# ...

selector = ModelSelector(
    models={
        planner_llm: ["gpt-4o-mini", "gpt-4o"],
        executor_llm: ["gpt-4o-mini", "gpt-4o"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
    invoke_fn=my_pipeline,
)
results = selector.select_best(parallel=True)
```

## How it works

- **Detection:** `LangChainAdapter.detect()` checks for LangChain's `AgentExecutor` or similar classes
- **Invocation:** Calls `executor.invoke(input_data)`
- **Token tracking:** Uses a `_TokenTrackingCallback` that intercepts LLM responses
