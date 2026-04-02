---
name: agentopt
description: Use this skill when optimizing LLM model combinations for an existing agent workflow with AgentOpt, especially when creating an offline evaluation dataset, choosing selector/concurrency settings, and exporting benchmark-ready artifacts.
metadata:
  short-description: Optimize agent model combos with offline eval
---

# AgentOpt Skill

## What this skill does

This skill tells an agent exactly how to run AgentOpt end-to-end for offline model selection:
- define an agent wrapper (`__init__` + `run`),
- build/clean a labeled offline dataset,
- choose a selector strategy and concurrency budget,
- run evaluation and export reproducible artifacts.

## How it works (mental model)

AgentOpt evaluates model **combinations** over a labeled dataset:
1. Create one model combo (e.g., planner=`gpt-4o-mini`, solver=`gpt-4.1`).
2. Instantiate the user agent with that combo.
3. Run all dataset samples, score with `eval_fn`, and track latency/tokens/cost.
4. Repeat for other combos (or a searched subset, depending on `method`).
5. Rank combinations by quality first, then latency/cost tie-breakers.

`parallel=True` with `max_concurrent=N` controls the total in-flight API budget across all combo/datapoint evaluations.

Use this skill when the user wants to:
- run model selection for an agent pipeline (single-step or multi-step),
- create or clean an offline evaluation dataset,
- tune `method` / `parallel` / `max_concurrent`,
- produce shareable benchmark outputs (CSV + best config + run metadata).

## Fast Workflow

1. **Wrap agent with AgentOpt contract**
2. **Create/validate offline dataset**
3. **Define candidate model space**
4. **Run selector with explicit concurrency budget**
5. **Export artifacts for review/submission**

---

## 1) Agent Contract (required)

AgentOpt expects:
- `__init__(self, models)` where `models` is a dict like `{"planner": "gpt-4o-mini", "solver": "gpt-4o"}`
- `run(self, input_data)` returning output for one datapoint

```python
class MyAgent:
    def __init__(self, models):
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        # call your framework here (OpenAI / LangChain / CrewAI / etc.)
        return {"answer": "..."}
```

No inheritance/base class is required (duck typing).

---

## 2) Offline Dataset Creation (required)

### Dataset shape AgentOpt enforces

Dataset must:
- support `len(dataset)` and `dataset[i]`,
- be non-empty,
- contain elements that unpack as `(input_data, expected_answer)`.

Canonical form:

```python
dataset = [
    ({"question": "What is 2+2?"}, "4"),
    ({"question": "Capital of France?"}, "Paris"),
]
```

### Recommended JSONL schema for offline evaluation

Use JSONL with one object per line:

```json
{"input": {"question": "What is 2+2?"}, "expected": "4", "id": "sample-0001"}
{"input": {"question": "Capital of France?"}, "expected": "Paris", "id": "sample-0002"}
```

### Build dataset from traces/logs (example)

```python
import json
from pathlib import Path

def load_offline_dataset(path: str, limit: int | None = None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Keep only labeled rows
            if "input" not in obj or "expected" not in obj:
                continue
            rows.append((obj["input"], obj["expected"]))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("No usable labeled rows found.")
    return rows
```

If your production traces do not contain ground truth, create labels first (human or rubric-based) before running AgentOpt.

---

## 3) Model Space Definition

`models` is a dict mapping node name -> candidate list.

```python
models = {
    "planner": ["gpt-4o-mini", "gpt-4.1"],
    "solver": ["gpt-4o-mini", "gpt-4.1"],
}
```

Candidate entries can be:
- model name strings, or
- prebuilt LLM/model instances (framework-dependent), as long as your agent wrapper consumes them correctly.

Keep node keys in `models` aligned with what your agent reads in `__init__(self, models)`.

---

## 4) Run Selection

Use `ModelSelector(...)` for method dispatch:

```python
from agentopt import ModelSelector

def eval_fn(expected, actual):
    text = actual.get("answer", str(actual)) if isinstance(actual, dict) else str(actual)
    return 1.0 if str(expected).lower() in text.lower() else 0.0

selector = ModelSelector(
    agent=MyAgent,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    method="auto",  # auto -> arm_elimination
)

results = selector.select_best(parallel=True, max_concurrent=40)
results.print_summary()
```

### Method selection guidance

- `auto` / `arm_elimination`: default for most users.
- `brute_force`: exhaustive baseline.
- `random`: cheap exploratory baseline (`sample_fraction`).
- `hill_climbing`: local search with restarts (`num_restarts`).
- `epsilon_lucb`: near-best identification (`epsilon`, `n_initial`).
- `threshold`: classify combos above/below target score (`threshold`, `n_initial`).
- `lm_proposal`: proposer LLM picks one combo first (then evaluates that combo).

### Concurrency semantics (important)

`max_concurrent` is the **total API call budget** across all combos + datapoints.

Internally AgentOpt splits it into:
- datapoint-level concurrency per combo (`dp_concurrent`),
- combo-level concurrency (`n_combo`),

such that:
- `n_combo * dp_concurrent <= max_concurrent`

So increasing `max_concurrent` raises total throughput, not just per-combo throughput.

---

## 5) Export Benchmark-Ready Artifacts

Always export at least:
- ranked results table (`CSV`),
- best combo config (`YAML`),
- run metadata (`JSON`) containing method + concurrency + dataset size.

```python
from datetime import datetime, timezone
import json

results.to_csv("artifacts/results.csv")
results.export_config("artifacts/best_config.yaml")

meta = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "method": "auto",
    "parallel": True,
    "max_concurrent": 40,
    "dataset_size": len(dataset),
    "selection_wall_time_seconds": results.selection_wall_time_seconds,
    "best_combo": results.get_best_combo(),
}
with open("artifacts/run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
```

For external benchmark submission/repro checks, include:
- commit SHA,
- environment (package versions),
- random seed (if applicable),
- exact dataset split ID/version.

---

## 6) Troubleshooting Checklist

- **`TypeError` on dataset**: ensure `(input, expected)` tuple elements and non-empty sequence.
- **No token/cost tracking**: ensure LLM calls happen through supported HTTP stack (AgentOpt tracker uses transport interception).
- **Too slow**: increase `max_concurrent`, reduce candidate space, or switch from `brute_force` to `auto`.
- **Unstable rankings**: increase dataset size and/or enforce deterministic prompts where possible.
- **High rerun cost**: use `LLMTracker(cache_dir=...)` for disk cache reuse.

---

## Repo Pointers

- Core API: `src/agentopt/__init__.py`
- Selector internals + concurrency split: `src/agentopt/model_selection/base.py`
- Brute force reference: `src/agentopt/model_selection/brute_force.py`
- Quickstart docs: `docs/getting-started/quickstart.md`
- Selector docs: `docs/api/selectors.md`
- End-to-end examples: `examples/*.py`
