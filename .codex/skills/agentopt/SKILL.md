---
name: agentopt
description: Use when the user wants to pick the best LLM model (or combination of models) for a multi-step agent pipeline by running offline evaluation, including when no eval dataset exists yet and one must be generated. Trigger phrases include "which model should I use", "benchmark my planner/solver", "compare GPT-4o vs GPT-4o-mini for my agent", "model selection for my agent", "offline eval of model combos", "generate an eval dataset".
metadata:
  short-description: Pick model combos and create offline eval data when needed
---

# AgentOpt

## When to use this skill

Trigger when the user asks to:
- choose among candidate LLMs for an agent with one or more LLM-calling steps (planner, solver, reviewer, etc.),
- benchmark an agent pipeline against a labeled dataset,
- tune a concurrency/cost budget for an evaluation run,
- produce shareable artifacts (ranked CSV, best config YAML) from a model-selection run.

Do not trigger for: online A/B testing, pure prompt optimization, single-model evaluation.

## Prerequisites

- Python `>=3.10` (declared in `pyproject.toml`).
- Install: `uv pip install agentopt-py` (or `pip install agentopt-py`).
- API keys in env for each provider the candidate models use (e.g. `OPENAI_API_KEY`).
- Preferred: labeled offline dataset as `(input_data, expected_answer)` pairs.
- If no dataset exists: generate a draft eval dataset with the workflow in "Create an eval dataset from scratch" before running model selection.

## Minimal end-to-end example

This is a complete, runnable script. Start here; customize below.

```python
from agentopt import ModelSelector

class MyAgent:
    def __init__(self, models):
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        # Call your LLM(s) here using self.planner_model / self.solver_model.
        # Return whatever eval_fn expects as `actual`.
        return {"answer": "..."}

dataset = [
    ({"question": "What is 2+2?"}, "4"),
    ({"question": "Capital of France?"}, "Paris"),
]

def eval_fn(expected, actual):
    text = actual.get("answer", str(actual)) if isinstance(actual, dict) else str(actual)
    return 1.0 if str(expected).lower() in text.lower() else 0.0

selector = ModelSelector(
    agent=MyAgent,
    models={
        "planner": ["gpt-4o-mini", "gpt-4.1"],
        "solver": ["gpt-4o-mini", "gpt-4.1"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
    method="auto",
)

results = selector.select_best(parallel=True, max_concurrent=40)
results.print_summary()
results.to_csv("artifacts/results.csv")
results.export_config("artifacts/best_config.yaml")
print("best:", results.get_best_combo(), "cost:", results.selection_cost)
```

A working fuller version lives at `examples/custom_agent_example.py`.

---

## Customization

### 1. Agent contract

AgentOpt instantiates the user's agent with a combo dict and calls `run` per datapoint. Duck-typed — no base class.

```python
class MyAgent:
    def __init__(self, models):           # models = {"planner": "gpt-4o-mini", ...}
        ...
    def run(self, input_data):            # returns anything eval_fn can score
        ...
```

Keep node keys in `models` (the dict) aligned with the keys your `__init__` reads.

### 2. Dataset

Required shape: indexable, non-empty, elements unpack as `(input_data, expected)`.

```python
dataset = [
    ({"question": "..."}, "expected-answer"),
    ...
]
```

JSONL loader (recommended for reproducibility):

```python
import json
from pathlib import Path

def load_offline_dataset(path, limit=None):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if "input" not in obj or "expected" not in obj:
            continue
        rows.append((obj["input"], obj["expected"]))
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No labeled rows found.")
    return rows
```

### 2a. Create an eval dataset from scratch

Use this when the user has no labeled eval dataset. The goal is not to create a perfect benchmark automatically; it is to create a usable first offline eval set that can be inspected, refined, and versioned.

Workflow:

1. Ask the user for the task contract:
   - what the agent is supposed to do,
   - input fields,
   - output format,
   - success criteria,
   - known hard cases.

2. Use an LLM to generate diverse candidate inputs:
   - include easy, medium, hard, and edge-case examples,
   - avoid near-duplicates,
   - include realistic failure modes,
   - create at least 30 examples for a smoke eval and 100+ for a more useful eval.

3. Use an LLM to generate a scoring rubric:
   - exact-match fields when possible,
   - semantic criteria when exact match is too strict,
   - disqualifying errors,
   - optional partial-credit scale from `0.0` to `1.0`.

4. Run a strong reference model or agent to obtain expected outputs:
   - use the strongest available model (for example Opus 4.6 if available, GPT-4.1/4o class, or the user's current best model),
   - store both the raw reference output and the normalized expected answer.

5. Iterate once on inputs and rubric:
   - remove trivial examples the reference model solves too easily,
   - add harder cases where outputs expose ambiguity,
   - tighten rubric criteria where scoring is unclear,
   - keep a small sample for human inspection.

6. Save JSONL artifacts:
   - `eval_dataset.jsonl` with `input`, `expected`, `rubric`, and `id`,
   - `dataset_manifest.json` with generator model, reference model, date, size, and labeling policy.

Recommended JSONL row:

```json
{"id":"synthetic-0001","input":{"question":"..."},"expected":"...","rubric":{"score_type":"semantic","criteria":["..."],"disqualifiers":["..."]},"source":"synthetic_llm_generated","reference_model":"best_available"}
```

Loader:

```python
def load_generated_eval(path, limit=None):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        rows.append((obj["input"], {"expected": obj["expected"], "rubric": obj.get("rubric", {})}))
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No generated eval rows found.")
    return rows
```

Rubric-aware `eval_fn` pattern:

```python
def eval_fn(expected_obj, actual):
    expected = expected_obj["expected"] if isinstance(expected_obj, dict) else expected_obj
    text = actual.get("answer", str(actual)) if isinstance(actual, dict) else str(actual)
    return 1.0 if str(expected).lower() in text.lower() else 0.0
```

Important safeguards:

- Mark generated datasets clearly as synthetic; do not present them as human-labeled benchmarks.
- Keep generated inputs, rubric, and reference outputs so the dataset can be audited.
- Prefer human review for a small sample before using results to make expensive model decisions.
- Do not let the same weak candidate model generate labels for its own evaluation.

### 3. Model space

`models` is `{node_name: [candidate, ...]}`. Cartesian product defines the combo space.

```python
models = {
    "planner": ["gpt-4o-mini", "gpt-4.1"],
    "solver":  ["gpt-4o-mini", "gpt-4.1"],
}
```

Candidates may be model-name strings or framework-specific LLM instances, as long as your agent wrapper consumes them.

### 4. Selector method — decision table

`ModelSelector(...)` is a factory function; `method=` picks the algorithm. Pass method-specific knobs as kwargs to `ModelSelector(...)`.

| method | when to use | key kwargs |
|---|---|---|
| `auto` | **Default.** Aliases `arm_elimination`. Start here. | — |
| `arm_elimination` | Best-arm identification with statistical elimination. | `n_initial`, `growth_factor`, `confidence` |
| `brute_force` | Small combo space; exhaustive baseline. | — |
| `random` | Cheap exploratory scan over a large space. | `sample_fraction`, `seed` |
| `hill_climbing` | Local search; combos have smooth structure. | `max_iterations`, `num_restarts`, `patience`, `seed`, `batch_size` |
| `epsilon_lucb` | Find any combo within ε of the best. | `epsilon`, `n_initial`, `confidence` |
| `threshold` | Classify combos above/below a quality bar. | `threshold`, `n_initial`, `confidence` |
| `matrix_ucb` | UCB over a node × model matrix; structured spaces. | see `docs/api/selectors.md` |
| `matrix_ucb_lrf` | Low-rank-factorization variant of matrix UCB. | see `docs/api/selectors.md` |
| `lm_proposal` | Ask a proposer LLM to pick a combo, then evaluate it. | `proposer_model`, `proposer_client`, `objective`, `dataset_preview_size`, `node_descriptions` |
| `bayesian` | Surrogate-model-guided search over medium/large spaces. | `batch_size`, `sample_fraction` |

Other `ModelSelector` kwargs: `model_prices` (custom token pricing), `tracker` (e.g. `LLMTracker(cache_dir=...)` for disk-cache reuse), `node_descriptions` (used by `lm_proposal`).

### 5. Concurrency

`select_best(parallel=True, max_concurrent=N)` — `max_concurrent` is the **total in-flight API call budget across all combos × datapoints**, not per-combo.

Internal split (see `src/agentopt/model_selection/base.py:1055-1090`):

```
dp_concurrent = min(max(max_concurrent, 1), batch_size)
n_combo       = max(max_concurrent, 1) // dp_concurrent
# guarantee: n_combo * dp_concurrent <= max_concurrent
```

Raise `max_concurrent` for more throughput; bound by provider rate limits and local runtime.

### 6. Results & export

`select_best` returns a `SelectionResults` with:

- `print_summary()` — ranked table to stdout.
- `to_csv(path)` — full ranking (accuracy, latency, tokens, cost per combo).
- `export_config(path)` — best combo as YAML, ready for production.
- `get_best_combo()` → `dict[str, str]`.
- `plot_pareto(path=None)` — quality-vs-cost Pareto front.
- `selection_cost` — total USD spent on the run.
- `selection_wall_time_seconds` — wall clock.

For run metadata:

```python
from datetime import datetime, timezone
import json

with open("artifacts/run_metadata.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "auto",
        "parallel": True,
        "max_concurrent": 40,
        "dataset_size": len(dataset),
        "selection_wall_time_seconds": results.selection_wall_time_seconds,
        "selection_cost_usd": results.selection_cost,
        "best_combo": results.get_best_combo(),
    }, f, indent=2)
```

For external benchmark submission packets (directory layout, manifest schema, README template), see `docs/benchmark-submission.md`.

---

## Verification

After a run, confirm:

1. The script exits 0 and `results.print_summary()` prints a ranked table with one row per evaluated combo.
2. `artifacts/results.csv` exists and has `> 1` data row (header + combos).
3. `artifacts/best_config.yaml` exists and parses as YAML with the expected node keys.
4. `results.get_best_combo()` returns a dict whose keys match the `models` dict.
5. `results.selection_cost` is a non-negative float; if zero on a non-cached run, token tracking is not wired up — see Troubleshooting.

## Troubleshooting

- **`TypeError` on dataset**: elements must unpack as `(input, expected)`; sequence must be non-empty.
- **No token/cost tracking**: LLM calls must go through a supported HTTP stack — `LLMTracker` uses transport interception.
- **Too slow**: raise `max_concurrent`, shrink the combo space, or switch `brute_force` → `auto`.
- **Unstable rankings**: grow the dataset; enforce deterministic prompts (temperature=0) where feasible.
- **Expensive reruns**: pass `tracker=LLMTracker(cache_dir=".agentopt_cache")` to `ModelSelector` for on-disk call cache.

## Repo pointers

- Public API: `src/agentopt/__init__.py`
- Selector base + concurrency split: `src/agentopt/model_selection/base.py`
- Method implementations: `src/agentopt/model_selection/`
- Quickstart: `docs/getting-started/quickstart.md`
- Selector reference: `docs/api/selectors.md`
- End-to-end examples: `examples/`
