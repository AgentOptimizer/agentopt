# Results

## SelectionResults

Returned by `selector.select_best()`. Contains all evaluation results.

| Method | Returns | Description |
|:-------|:--------|:------------|
| `print_summary()` | `None` | Print formatted table with rank, accuracy, latency, price |
| `get_best()` | `ModelResult` | Highest accuracy result (ties broken by latency) |
| `get_best_combo()` | `Dict[str, str]` | Best combination as `{"node": "model_name"}` |
| `to_csv(path)` | `None` | Export all results to CSV |
| `export_config(path)` | `None` | Export best combination as YAML config |

### Example

```python
results = selector.select_best(parallel=True)

# Print ranked table
results.print_summary()

# Access the winner
best = results.get_best()
print(f"Best: {best.model_name}, accuracy={best.accuracy:.1%}")

# Export
results.to_csv("all_results.csv")
results.export_config("optimized_config.yaml")
```

---

## ModelResult

Each evaluated combination produces a `ModelResult`:

| Field | Type | Description |
|:------|:-----|:------------|
| `model_name` | `str` | Combination label (e.g., `"planner=gpt-4o + solver=gpt-4o-mini"`) |
| `accuracy` | `float` | Mean eval score across all datapoints |
| `latency_seconds` | `float` | Mean latency per datapoint |
| `input_tokens` | `Dict[str, int]` | Input tokens by model |
| `output_tokens` | `Dict[str, int]` | Output tokens by model |
| `estimated_price` | `float` | Estimated cost in USD |
| `is_best` | `bool` | Whether this is the top-ranked combination |
| `datapoint_results` | `List[DatapointResult]` | Per-datapoint breakdown |

---

## DatapointResult

Per-datapoint evaluation detail:

| Field | Type | Description |
|:------|:-----|:------------|
| `datapoint_index` | `int` | Index in the dataset |
| `datapoint_id` | `str` | Unique identifier |
| `score` | `float` | Eval score for this datapoint |
| `latency_seconds` | `float` | Latency for this datapoint |
