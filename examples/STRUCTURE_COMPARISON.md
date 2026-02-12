# Baseline vs AgentOpt Structure (OpenAI & Claude SDKs)

This doc shows the before/after wiring so you can see exactly what changed when adding AgentOpt.

## OpenAI SDK

### Baseline (e.g., `examples/openai_sdk/math_qa_baseline.py`)
- Components: `OpenAI()` client, one call to `chat.completions.create(...)` per question.
- Data flow: `question` → `chat.completions.create(model="gpt-4o-mini", ...)` → `answer`.
- No model swapping, no wrappers, no rebuilding.

### AgentOpt (simple wrapper, e.g., `math_qa_agentopt.py`)
- Added: `OpenAIChat` wrapper (mutable `model`, `.invoke`), `ModelProxy` around it, `ModelSelector` driving evaluation.
- Data flow: `question` → `ModelSelector` → `proxy.invoke` → same `chat.completions.create`, but `proxy.model` changes across candidates.
- Touch points: Only the model string mutates; request/response logic is unchanged.

### AgentOpt (Runner/Agents pattern, `agents_runner_example.py`)
- Added: `build_agent(model)` (same as your vanilla agent factory), `run_agent(agent, question)` (Runner API), `AgentFactoryRunner(proxy, build_agent, run_agent)` exposing `.invoke`.
- Data flow: `question` → `ModelSelector` → `runner.invoke` → rebuild agent with current model → `agents.runs.create_and_poll` → `answer`.
- Touch points: Only the `model` passed to the factory changes; tools/prompt remain the same.

## Claude SDK

### Baseline (e.g., `examples/claude_sdk/math_qa_baseline.py`)
- Components: `Anthropic()` client, one call to `messages.create(...)` per question.
- Data flow: `question` → `messages.create(model="claude-3-5-haiku-latest", ...)` → `answer`.

### AgentOpt (simple wrapper, e.g., `math_qa_agentopt.py`)
- Added: `ClaudeChat` wrapper + `ModelProxy`, `ModelSelector` with `invoke_fn=lambda payload: proxy.invoke(payload)`.
- Data flow: `question` → `ModelSelector` → `proxy.invoke` → same `messages.create`, with `proxy.model` swapped per candidate.

### AgentOpt (Messages pattern, `agents_runner_example.py`)
- Added: `build_agent(model)` returning a config with `model` + prompt/tools, `run_agent(agent_cfg, question)` calling `messages.create`, `AgentFactoryRunner(proxy, build_agent, run_agent)`.
- Data flow: `question` → `ModelSelector` → `runner.invoke` → rebuild config with current model → `messages.create` → `answer`.

## What stayed the same
- Baseline request/response code paths are unchanged; AgentOpt only inserts `ModelProxy`/`AgentFactoryRunner` to swap model names.
- Tools, prompts, and task logic are reused verbatim from the baseline/vanilla factories.

## Quick map of key files
- Baselines: `examples/openai_sdk/*_baseline.py`, `examples/claude_sdk/*_baseline.py`
- Simple AgentOpt: `examples/openai_sdk/*_agentopt.py`, `examples/claude_sdk/*_agentopt.py`
- Runner/Factory pattern: `examples/openai_sdk/agents_runner_example.py`, `examples/claude_sdk/agents_runner_example.py`
- Shared utilities: `examples/sdk_shared.py`
- High-level summary: `examples/SDK_OVERVIEW.md`
