# Agent Instructions

## Repository Scope

- AgentOpt is the target and main repository. Implement requested features in `src/agentopt`, `tests`, `examples`, and docs as appropriate.
- Treat `cortex/` as read-only reference material. Do not edit files under `cortex/`; use them only to understand behavior, APIs, or implementation ideas that may inform AgentOpt changes.
- When a requested feature is inspired by Cortex, add all necessary functionality to AgentOpt itself. Do not rely on Cortex files at runtime.
- Move as little code from Cortex as necessary. Prefer a clean AgentOpt-native implementation from scratch when that avoids unnecessary coupling or code debt.

## Python Workflow

- Use `uv` from the repository root for Python commands. The repo declares Python `>=3.10`, and `.python-version` is `3.11`.
- Set up the development environment with:

  ```bash
  uv sync --extra dev
  ```

- Run Python through the project environment:

  ```bash
  uv run python ...
  ```

- Run tests with:

  ```bash
  uv run pytest
  ```

- When examples or optional integrations need dependencies, install the relevant extras with `uv sync --extra dev --extra <extra-name>` instead of using ad hoc `pip install`.
- Add or change project dependencies in `pyproject.toml`; do not depend on packages installed outside the `uv` environment.

## Execution Guardrails

- Do not run tests, examples, benchmark scripts, or ad hoc commands that call LLM stages or external LLM APIs unless the user explicitly asks for that run. Prefer dry-run modes, syntax checks, and non-LLM unit tests when validating changes.
- When the user explicitly permits LLM-backed runs in a session, keep runs narrowly scoped and report the approximate LLM spend in the response. Estimate spend from the model used, provider pricing, and rough input/output token counts. Make clear when the amount is an estimate.
