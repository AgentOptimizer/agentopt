# Claude SDK + AgentOpt

This folder mirrors the OpenAI example but for Anthropic Claude Messages.

## Files
- `math_qa_baseline.py` / `summary_baseline.py`: vanilla Claude calls.
- `math_qa_agentopt.py` / `summary_agentopt.py`: ModelSelector + ModelProxy swapping models.
- `agents_runner_example.py`: general pattern using a simple `build_agent` config and `run_agent`.

## How AgentOpt fits
1) `ClaudeChat` (or the `build_agent` SimpleNamespace) exposes a mutable `model` and `invoke` behavior.
2) `ModelProxy` keeps that wrapper stable while ModelSelector swaps model names during evaluation.
3) `invoke_fn` rebuilds the Claude message config per call so tools/prompt stay untouched.

## Next steps (minimal changes)
- Replace the placeholder prompt/tools in `agents_runner_example.py` with your real agent setup.
- Keep the `model` string the only thing that changes; avoid touching the rest of your agent wiring.
- Point `dataset_path` to your JSONL task set; no changes to agent code required.
- Optionally share a single `Anthropic()` client if you want less per-call overhead.
