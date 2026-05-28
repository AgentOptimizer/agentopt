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

Top-level fields: `results`, `selection_wall_time_seconds`, `selection_cost` (USD; `None` when pricing is unavailable).

### Example

```python
results = selector.select_best(parallel=True)
results.print_summary()

best = results.get_best()
print(f"Best: {best.model_name}, accuracy={best.accuracy:.1%}, ${best.price:.6f}/sample")

results.to_csv("all_results.csv")
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
| `attribute` | `str` | Metric track the result was scored under (algorithms like `threshold` produce multiple). |
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
