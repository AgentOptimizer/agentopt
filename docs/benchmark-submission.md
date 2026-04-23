# Benchmark submission packets

This doc describes a recommended layout for sharing AgentOpt model-selection results with an external benchmark, collaborator, or reviewer. It is optional — use it when you need reproducible artifacts that a third party can re-run or audit.

## Layout

One directory per benchmark run:

```text
artifacts/
  <benchmark_name>/<run_id>/
    results.csv
    best_config.yaml
    run_metadata.json
    dataset_manifest.json
    README_submission.md
```

## File contents

### `results.csv`

Produced by `results.to_csv(...)` — full ranked table (accuracy, latency, tokens, cost per combo).

### `best_config.yaml`

Produced by `results.export_config(...)` — the winning combo, ready to consume in production.

### `run_metadata.json`

Captures the selector configuration. Minimum fields:

```json
{
  "timestamp_utc": "2026-04-15T00:00:00Z",
  "method": "auto",
  "parallel": true,
  "max_concurrent": 40,
  "dataset_size": 120,
  "selection_wall_time_seconds": 184.2,
  "selection_cost_usd": 1.87,
  "best_combo": {"planner": "gpt-4o-mini", "solver": "gpt-4.1"},
  "commit_sha": "abc1234",
  "package_versions": {"agentopt": "0.1.0", "openai": "1.x"},
  "seed": 42
}
```

Include `commit_sha`, `package_versions`, and `seed` whenever external reproducibility matters.

### `dataset_manifest.json`

Describes dataset provenance and labeling policy:

```json
{
  "name": "my_offline_eval_v1",
  "num_samples": 120,
  "format": "jsonl(input, expected)",
  "split": "offline_eval",
  "label_policy": "human_verified",
  "created_at_utc": "2026-04-02T00:00:00Z"
}
```

### `README_submission.md`

Short human-readable summary. State:

- benchmark / task name,
- method and key parameters (`parallel`, `max_concurrent`, method-specific kwargs),
- best combo and headline metrics,
- exact reproducibility commands (install, env vars, run script).

## Submission channels

- **GitHub repo or PR:** attach the packet in a PR or issue comment and link the commit SHA.
- **External benchmark portal:** upload the packet unchanged.

Do not edit the contents of the packet between generation and submission — the manifest and metadata are the audit trail.
