# Results

## SelectionResults

Returned by `selector.select_best()`. Holds every evaluated combination and the metadata needed to inspect, export, and compare them.

| Method | Returns | Description |
|:---|:---|:---|
| `print_summary()` | `None` | Print a ranked table with accuracy, latency, tokens, and price. When any result has `combined_objective` set, the table sorts by that value. |
| `get_best(attribute=None)` | `ModelResult?` | The `is_best` combination. Pass `attribute` to scope the lookup to a single metric track. |
| `get_best_combo()` | `Dict[str, str]?` | Best combination as `{"node": "model_name"}`. |
| `get_by_attribute(attribute)` | `List[ModelResult]` | All results for a given attribute. |
| `to_csv(path)` | `None` | Export every result to CSV. |
| `export_config(path, api_key_env_vars=None)` | `None` | Export the best combination as a LiteLLM-style YAML config; `api_key_env_vars` overrides per-provider env-var names. |

Iterable: `for result in results: ...` yields `ModelResult`s.

Top-level fields are available on both live snapshots and the final result:

| Field | Meaning |
|:---|:---|
| `results` | Observed configuration results, including partial evaluations where the method supports them. |
| `is_final` | `False` for progress snapshots; `True` for the final callback and returned result. |
| `stop_reason` | Current phase or final stopping reason. A user-requested stop reports `user_stopped`; UCB reports `evaluation_failed` when an entire acquisition batch produces no observation. |
| `total_evaluations` | Successful configuration–question evaluations represented in the results. Failed evaluations may incur `selection_cost` without increasing this count. |
| `selection_wall_time_seconds` | Elapsed selection time at the snapshot. |
| `selection_cost` | New non-cached search spending in USD; `None` when pricing is unavailable. |

Pass `on_update` to `select_best()` to receive independent snapshots.
`selector.latest_results` returns a copy of the latest snapshot, or `None`
before the first update. Observations can appear before a recommendation is
eligible: Gittins needs a complete warm start, and random selection needs a
complete configuration or shared-question sweep. A current recommendation can
change as more data arrives. See [live selection](selectors.md#live-recommendations-and-user-stopping)
for stopping and update boundaries.

### GittinsSelectionResults

Gittins returns a `SelectionResults` subclass with these additional fields:

| Field or method | Meaning |
|:---|:---|
| `objective_mode` | `"single"` or `"multi"`. |
| `recommendations` / `get_recommendations()` | Recommended configurations at the snapshot, with predicted accuracy/cost, standard deviations, conservative coordinates, node-to-model `combo`, and observed sample count. Single mode defaults to zero uncertainty penalty; multi mode defaults to one standard deviation. |
| `stop_reason` | Final reasons include `user_stopped`, `question_budget`, `search_cost_budget`, `warm_start_budget`, `all_arms_completed`, `gittins_stop` (single), and `direction_eta_numerical_floor` (multi). Live snapshots can report `warming_up` or `running`. |
| `total_evaluations` | Successful configuration–question evaluations, including warm start. |
| `search_cost_overshoot_usd` | Actual non-cached spend above the optional soft dollar cap. |
| `direction_eta` | Current multiplier of each search direction. Single mode reports `[1.0]`; its separate `cost_scaling_factor` stays fixed. |
| `choose(max_cost_usd=..., min_score=..., cost_weight=...)` | Select a candidate using conservative deployment criteria; returns a recommendation or `None` if infeasible. |

`ModelResult.accuracy`, `price`, and `combined_objective` describe observed
data. Recommendation `accuracy` and `cost_usd` predict the supplied dataset's
full mean; `objective` and `conservative_objective` describe the scalar
prediction in single mode. The two kinds of values must not be conflated.
Single mode marks one `is_best` when a candidate is eligible. Multi mode
marks none: use the recommendation set and a deployment preference instead.
See [Gittins](selectors.md#gittins-single-and-multi-objective) for assumptions,
budget guards, and Gaussian uncertainty semantics.

### RandomSelectionResults

Random baselines expose the same recommendation and `choose(...)` interface, plus
`sampling` and `completed_units`. Their score/cost values are empirical means from
complete configurations or shared-question sweeps. Conservative fields equal the
means; zero std fields mean no uncertainty model was fitted. Partial units are
retained in evaluation records but excluded from recommendations. In single
mode, `get_best()` and `get_best_combo()` identify an eligible empirical winner;
in multi mode, use `get_recommendations()` and `choose(...)`.

### Single-objective example

```python
results = selector.select_best(parallel=True)
results.print_summary()

best = results.get_best()
if best is not None:
    print(f"Best: {best.model_name}, accuracy={best.accuracy:.1%}, ${best.price:.6f}/sample")

results.to_csv("all_results.csv")
if best is not None:
    results.export_config("optimized_config.yaml")
```

---

## ModelResult

One per evaluated combination.

| Field | Type | Description |
|:---|:---|:---|
| `model_name` | `str` | Combination label, e.g. `"planner=gpt-4o + solver=gpt-4o-mini"`. |
| `accuracy` | `float` | Mean eval score across evaluated datapoints. |
| `combined_objective` | `float?` | Mean per-datapoint combined score when `lambda_cost` and/or `lambda_latency` were set on the selector; `None` otherwise. See [selectors.md — Combined objective](selectors.md#combined-objective-optional-costlatency-weights). |
| `latency_seconds` | `float` | Mean latency per datapoint. |
| `input_tokens` | `Dict[str, int]` | Input tokens by model. |
| `output_tokens` | `Dict[str, int]` | Output tokens by model. |
| `attribute` | `str` | Metric track the result was scored under. |
| `is_best` | `bool` | Whether this is the top-ranked combination. |
| `datapoint_results` | `List[DatapointResult]` | Per-datapoint breakdown. |

Properties:

| Property | Returns | |
|:---|:---|:---|
| `total_input_tokens` | `int` | Sum across models. |
| `total_output_tokens` | `int` | Sum across models. |
| `price` | `float?` | Per-sample USD cost, or `None` if pricing for any used model is unavailable. |
| `num_samples` | `int` | `len(datapoint_results)`, with `1` as the fallback for failed combos. |

`str(result)` returns a one-line "name (accuracy: X%, latency: Ys, tokens: {…}, price: $…)" summary.

---

## DatapointResult

Per-datapoint detail inside `ModelResult.datapoint_results`.

| Field | Type | Description |
|:---|:---|:---|
| `datapoint_index` | `int` | Index in the dataset. |
| `score` | `float` | Eval score. |
| `latency_seconds` | `float` | Latency for this datapoint. |
| `input_tokens` | `Dict[str, int]` | Input tokens by model. |
| `output_tokens` | `Dict[str, int]` | Output tokens by model. |
