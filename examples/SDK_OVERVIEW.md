# SDK Examples Overview

This document summarizes how the OpenAI and Claude SDK examples are structured and how AgentOpt is layered on top with minimal intrusion.

## Shared building blocks
- `examples/sdk_shared.py`
  - **Datasets:** `load_jsonl_dataset` → list of `({"input": ...}, expected)` tuples; `small_summary_dataset` for tiny inline cases.
  - **Eval:** `eval_fn(expected, actual)` simple containment check (swap in your own scorer as needed).
  - **Wrappers:** `OpenAIChat`, `ClaudeChat` expose a mutable `model` attribute and `.invoke` so ModelProxy can swap models.
  - **Adapter:** `AgentFactoryRunner` gives ModelSelector an `.invoke` without you writing a custom `invoke_fn`; it rebuilds the agent/config using the current model string from ModelProxy and calls your `run_fn`.

## OpenAI SDK

### Baselines (no AgentOpt)
- `examples/openai_sdk/math_qa_baseline.py`
- `examples/openai_sdk/summary_baseline.py`
Structure: instantiate `OpenAI()`, call `chat.completions.create(...)`, print outputs. No model swapping.

### AgentOpt versions
- `examples/openai_sdk/math_qa_agentopt.py`
- `examples/openai_sdk/summary_agentopt.py`
Structure: wrap `OpenAIChat` with `ModelProxy`; `ModelSelector` iterates candidate models using `invoke_fn=lambda payload: proxy.invoke(payload)`.

### General Runner pattern (Agents SDK)
- `examples/openai_sdk/agents_runner_example.py`
  - `build_agent(model)`: your normal Agents SDK factory (tools/prompt unchanged).
  - `run_agent(agent, question)`: executes via `client.agents.runs.create_and_poll` (Runner API).
  - `ModelProxy(SimpleNamespace(model="gpt-4o-mini"))` holds the current model string.
  - `AgentFactoryRunner(proxy, build_agent, run_agent)` provides `.invoke` so `ModelSelector(..., agent=runner)` can swap models without a custom `invoke_fn`.

## Claude SDK

### Baselines (no AgentOpt)
- `examples/claude_sdk/math_qa_baseline.py`
- `examples/claude_sdk/summary_baseline.py`
Structure: call `Anthropic().messages.create(...)`, print outputs. No model swapping.

### AgentOpt versions
- `examples/claude_sdk/math_qa_agentopt.py`
- `examples/claude_sdk/summary_agentopt.py`
Structure: wrap `ClaudeChat` with `ModelProxy`; `ModelSelector` iterates models via `invoke_fn=lambda payload: proxy.invoke(payload)`.

### General Messages pattern
- `examples/claude_sdk/agents_runner_example.py`
  - `build_agent(model)`: returns a config (SimpleNamespace) with `model` and your prompt/tools (if any).
  - `run_agent(agent_cfg, question)`: calls `Anthropic().messages.create(...)` once.
  - `ModelProxy(SimpleNamespace(model="claude-3-5-haiku-latest"))` + `AgentFactoryRunner(...)` give ModelSelector an `.invoke` entrypoint with no custom `invoke_fn`.

## How AgentOpt stays minimally invasive
1) Keep your existing agent factory and tool/prompt wiring unchanged.
2) Introduce a `ModelProxy` that only holds the mutable `model` field.
3) Plug either:
   - `AgentFactoryRunner(proxy, factory, run_fn)` into `agent=` (preferred), or
   - `invoke_fn=lambda payload: proxy.invoke(payload)` when using the simple wrappers.
4) Provide `models={proxy: ["model-a", "model-b"]}`, `eval_fn`, and `dataset` to `ModelSelector`.
5) After `results.get_best()`, rebuild your agent once with `best.model_name` for production.

## Quickstart commands
- Set keys: `export OPENAI_API_KEY=...` (and `ANTHROPIC_API_KEY=...` for Claude).
- Run a baseline: `python examples/openai_sdk/math_qa_baseline.py`
- Run AgentOpt (simple wrapper): `python examples/openai_sdk/math_qa_agentopt.py`
- Run Runner pattern: `python examples/openai_sdk/agents_runner_example.py`
- Run convenience batch: `python scripts/rest_examples.py`

## Customize
- Swap in your own tools/prompts inside the `build_agent` functions.
- Replace `eval_fn` with task-specific scoring.
- Point `dataset_path` to your JSONL; keep the `(input_dict, expected)` tuple shape.
