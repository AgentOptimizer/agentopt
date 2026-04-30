# BIRD NL2SQL Benchmark

This directory contains AgentOpt-native setup instructions for the BIRD dev benchmark. The dataset is large, so downloaded/copied data lives under `benchmarks/bird/data/` and is ignored by git.

## Set Up Data

Download the official dev archive:

```bash
uv run python benchmarks/bird/setup_bird.py
```

If this workspace already has the Cortex reference checkout with BIRD data, copy it without modifying Cortex:

```bash
uv run python benchmarks/bird/setup_bird.py --from-cortex
```

You can also point at an existing BIRD dev directory:

```bash
uv run python benchmarks/bird/setup_bird.py --source-dir /path/to/dev_20240627
```

Expected local layout:

```text
benchmarks/bird/data/
  dev.json
  dev.sql
  dev_tables.json
  dev_databases/
    <db_id>/<db_id>.sqlite
```

## Quick NL2SQL Check

Print the prompt without calling an LLM:

```bash
uv run python examples/bird_nl2sql_langgraph.py --question-id 0 --dry-run
```

Run the simple LangGraph NL2SQL example. This calls the configured LLM provider:

```bash
OPENAI_API_KEY=... uv run python examples/bird_nl2sql_langgraph.py --question-id 0 --model gpt-4o-mini
```

The example uses the same model for initial SQL generation and refinement. By default it allows one refinement after the initial attempt. Increase or disable that with:

```bash
uv run python examples/bird_nl2sql_langgraph.py --question-id 0 --max-refinements 2
uv run python examples/bird_nl2sql_langgraph.py --question-id 0 --max-refinements 0
```

Refinement always triggers on SQL execution errors. When evaluation is enabled, benchmark result mismatches also trigger generic repair feedback unless `--no-refine-on-mismatch` is set.

Install optional dependencies first if needed:

```bash
uv sync --extra dev --extra langgraph
```
