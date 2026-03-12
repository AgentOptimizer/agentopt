# Framework Support

AgentOpt works with all major LLM agent frameworks through its [adapter architecture](../concepts/adapters.md).

## Support matrix

| Framework | Proxy works directly? | Invoke method | Cross-provider? | Multi-agent | Parallel |
|-----------|----------------------|---------------|-----------------|-------------|----------|
| [CrewAI](crewai.md) | Yes (duck typing) | `.kickoff()` | Yes | Yes | Yes (adapter) |
| [LangChain](langchain.md) | Yes (duck typing) | `.invoke()` | Yes | Yes | Yes (adapter) |
| [LangGraph](langgraph.md) | N/A (uses `invoke_fn`) | graph `.invoke()` | Yes | Yes | Yes (thread-local) |
| [LlamaIndex](llamaindex.md) | No (Pydantic strict) | `.run()` (async) | Yes | Yes | Yes (adapter) |
| [OpenAI SDK](openai-sdk.md) | Yes (ABC virtual subclass) | `Runner.run_sync()` | OpenAI only | Yes | Yes |
| [Claude SDK](claude-sdk.md) | N/A (uses `invoke_fn`) | `query()` (async) | Claude only | Yes | Yes (thread-local) |
| [AG2](ag2.md) | No (patched validation) | `.run()` | OpenAI + Anthropic | Yes | Yes |
| [Custom](custom.md) | Yes | User-defined | Yes | Yes | Yes (thread-local) |

## Choosing an approach

**Use `agent=`** when your framework has a standard agent object (CrewAI Crew, LangChain AgentExecutor, LlamaIndex AgentWorkflow, OpenAI SDK Agent, AG2 ConversableAgent). AgentOpt auto-detects the framework and handles invocation and cloning.

**Use `invoke_fn=`** when you need custom control — for LangGraph compiled graphs, Claude SDK's functional API, or any custom pipeline.

## Known limitations

- **OpenAI SDK** only supports OpenAI models natively
- **Claude SDK** only supports Claude models (uses short aliases: `"haiku"`, `"sonnet"`, `"opus"`)
- **AG2** supports OpenAI and Anthropic models; other providers not yet supported
- **OpenRouter** has compatibility issues with some frameworks; prefer native API keys
