# OpenAI SDK + AgentOpt

This folder shows how to wrap the OpenAI Agents SDK with AgentOpt while keeping the agent’s tools and logic unchanged.

## Files
- `math_qa_baseline.py` / `summary_baseline.py`: vanilla SDK runs.
- `math_qa_agentopt.py` / `summary_agentopt.py`: ModelSelector swaps models via ModelProxy.
- `agents_runner_example.py`: general pattern using `build_agent` + `run_agent` with Runner.

## How AgentOpt fits
1) `OpenAIChat` (or the Runner example) exposes a mutable `model` field and an `invoke` function.
2) `ModelProxy` wraps that object so the model name can be swapped without rebuilding tools.
3) `ModelSelector` iterates candidate models, calling `invoke_fn` to execute the agent per dataset row.

## Next steps (minimal changes)
- Replace placeholder tools/prompt in `agents_runner_example.py` with real tools.
- If using the new Responses/Runner API, keep the wrapper stable and only pass different `model` strings.
- Add a small JSONL dataset under `examples/datasets/` to match your task; no agent code changes needed.
- Optionally cache clients externally if you want to avoid re-instantiation per eval.
