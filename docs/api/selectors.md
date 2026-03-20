# Selectors

All selectors share a common constructor interface and the `select_best()` method.

## Common Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `agent_fn` | `Callable` | Factory: `(models_dict) -> callable_agent` |
| `models` | `Dict[str, List]` | Maps node names to candidate model lists |
| `eval_fn` | `Callable` | `(expected, actual) -> float` score in `[0, 1]` |
| `dataset` | `List[Tuple]` | `[(input_data, expected_answer), ...]` |
| `invoke_fn` | `Callable`, optional | Custom `(agent, input) -> result`. Default: `agent(input)` |
| `model_prices` | `Dict`, optional | Custom pricing: `{"model": {"input_price": x, "output_price": y}}` |
| `tracker` | `LLMTracker`, optional | Custom tracker instance (e.g., with disk cache) |

## `select_best()`

```python
results = selector.select_best(
    parallel=False,       # Use async parallel evaluation
    max_concurrent=20,    # Max concurrent API calls per combination
)
```

Returns a [`SelectionResults`](results.md) object.

!!! note "Automatic cleanup"
    `select_best()` automatically calls `tracker.stop()` when it returns (or raises), flushing any cached data to disk.

---

## Selector Classes

::: agentopt.model_selection.brute_force.BruteForceModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.random_search.RandomSearchModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.hill_climbing.HillClimbingModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.arm_elimination.ArmEliminationModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.hyperband.HyperbandModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.lm_proposal.LMProposalModelSelector
    options:
      members: false
      show_bases: false
