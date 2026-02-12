# OpenAI Agents SDK: Baseline vs AgentOpt

This folder shows the plain OpenAI Agents SDK flow (no AgentOpt) and the minimal additions needed to let AgentOpt swap models.

## Baseline structure (math_qa_baseline.py)
- Build an Agents SDK `Agent` with `name`, `model`, `instructions`.
- Execute each question with `Runner.run_sync(agent, prompt)`.
- No wrappers, no model swapping.

## AgentOpt structure (math_qa_agentopt.py)
- `build_agent(model)` stays the same.
- `ModelProxy` holds the mutable `model` value.
- `AgentFactoryRunner` rebuilds the Agent for each candidate model and calls `Runner.run_sync`.
- `ModelSelector` receives `{proxy: [model_a, model_b]}`, the dataset, and `eval_fn`; only the model string changes.

## Runner pattern (agents_runner_example.py)
- Same as AgentOpt above, but organized as a reusable pattern (factory + runner wrapped by AgentFactoryRunner).

## Notes
- Dataset loader/eval live in `utils.py` in this folder (no shared files).
- Swap in your own tools/prompts inside `build_agent`; AgentOpt wiring remains unchanged.
