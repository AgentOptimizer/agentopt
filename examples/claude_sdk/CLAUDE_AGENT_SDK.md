# Claude Agent SDK: Baseline vs AgentOpt (async query API)

This folder uses the new Claude Agent SDK `query`/`ClaudeAgentOptions` pattern and shows the minimal additions for AgentOpt model swapping.

## Baseline structure (math_qa_baseline.py)
- Build a `ClaudeAgentOptions(model=...)`.
- Call `run_query_sync(prompt, model)` which wraps the async `claude_agent_sdk.query(...)` iterator and returns the final result.
- No wrappers, no model swapping.

## AgentOpt structure (math_qa_agentopt.py)
- `ModelProxy` holds the mutable `model` value (a `ClaudeAgentOptions.model` string).
- `AgentFactoryRunner` rebuilds a `ClaudeAgentOptions` per candidate and calls `run_query_sync`.
- `ModelSelector` receives `{proxy: [model_a, model_b]}`, the dataset, and `eval_fn`; only the model string changes.

## Runner pattern (agents_runner_example.py)
- Same query-based flow, packaged as a reusable factory/runner wrapped with `AgentFactoryRunner` for ModelSelector.

## Notes
- Dataset loader/eval/runner helpers live in `utils.py` in this folder (no shared files).
- Enable tools or other options by passing them through `run_query_sync(..., allowed_tools=[...])` if needed; the AgentOpt wiring stays the same.
