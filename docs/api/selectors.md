# Selectors

A `ModelSelector` picks the best model **combination** for an agent against a dataset: instantiate the agent with each candidate dict, evaluate, rank by `eval_fn`. All selectors share one constructor surface and one entry point — `select_best()` — differing only in their search algorithm.

```python
from agentopt import ModelSelector

selector = ModelSelector(
    agent=MyAgent,
    models={"planner": ["gpt-4o", "gpt-4o-mini"], "solver": ["gpt-4o-mini"]},
    eval_fn=lambda expected, actual: float(actual == expected),
    dataset=[(inp, expected), ...],
    method="auto",                # arm_elimination — strong + cheap
)
results = selector.select_best(parallel=True, max_concurrent=20)
results.print_summary()
```

## Common parameters

| Parameter | Type | Description |
|:---|:---|:---|
| `agent` | `type` | Agent class with `__init__(self, models)` and `run(self, input_data)`. Duck-typed — no base class required. |
| `models` | `Dict[str, List]` | Maps node names to candidate model lists (e.g. `{"planner": ["gpt-4o", "gpt-4o-mini"]}`). |
| `eval_fn` | `Callable` | `(expected, actual) -> float` score (higher is better). |
| `dataset` | `Sequence[Tuple]` | `[(input_data, expected_answer), ...]`. |
| `model_prices` | `Dict`, optional | Custom pricing overrides: `{"model": {"input_price": x, "output_price": y}}` in $/MTok. Required for cost terms when `lambda_cost > 0`. |
| `lambda_cost` | `float`, optional | Weight on **normalized** per-sample cost in the combined objective. Default `0.0` (disabled). See [Combined objective](#combined-objective-optional-costlatency-weights) below. |
| `lambda_latency` | `float`, optional | Weight on **normalized** per-sample latency in the combined objective. Default `0.0` (disabled). |
| `node_descriptions` | `Dict[str, str]`, optional | Human-readable descriptions per node — surfaced in `LMProposalModelSelector`. |
| `tracker` | `LLMTracker`, optional | Bring your own. Defaults to a fresh `LLMTracker()` started in the constructor. Pass one in to share a cache across runs, route via a daemon (`AGENTOPT_GATEWAY_URL`), or post-process records after `select_best()` returns. |

The selector calls `tracker.start()` in the constructor and `tracker.stop()` when `select_best()` returns or raises. Record queries on the tracker remain valid after `stop()`, so post-run analysis works:

```python
tracker = LLMTracker(cache_dir="./shared_cache")
selector = ModelSelector(..., tracker=tracker)
selector.select_best()
print(tracker.get_usage())          # tracker.stop() already called; records still here
```

See [tracker.md](tracker.md) for the full tracker surface.

## Combined objective (optional cost/latency weights)

By default, selectors optimize **`eval_fn` score only** (typically accuracy) and break ties with latency, then price. To trade accuracy against cost and latency in one scalar reward, pass optional weights on the constructor (or via `ModelSelector(..., **kwargs)`):

| Parameter | Default | Effect |
|:---|:---|:---|
| `lambda_cost` | `0.0` | Penalizes normalized per-sample **token cost** (USD from the tracker, or `model_prices`). |
| `lambda_latency` | `0.0` | Penalizes normalized per-sample **wall-clock latency** (seconds). |

Omit both parameters (or leave them at `0.0`) for the original accuracy-centric behavior. Set one or both when you want multi-metric selection.

### Formula

For each datapoint, after observations are recorded:

```
combined = score
         - lambda_cost    * norm(cost)
         - lambda_latency * norm(latency)
```

- **`score`** — return value of `eval_fn` (higher is better).
- **`norm(·)`** — min–max scale to `[0, 1]` using running min/max over **all** samples seen during that selector run (updated as more combos are evaluated).
- **Per combination** — mean of per-datapoint combined values → `ModelResult.combined_objective` (see [results.md](results.md)).

This is a **linear scalarization**, not Pareto exploration. Larger `lambda_*` penalize cost/latency more strongly relative to score.

### Example

```python
selector = ModelSelector(
    agent=MyAgent,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    method="matrix_ucb",
    lambda_cost=0.3,      # optional — omit for accuracy-only
    lambda_latency=0.2,
    model_prices={        # recommended when lambda_cost > 0
        "gpt-4o": {"input_price": 2.5, "output_price": 10.0},
        "gpt-4o-mini": {"input_price": 0.15, "output_price": 0.6},
    },
)
results = selector.select_best(parallel=True)
results.print_summary()   # ranks by combined_objective when lambdas are set
```

### How each method uses the weights

| Methods | During search | Final `is_best` |
|:---|:---|:---|
| `matrix_ucb`, `matrix_ucb_lrf` | UCB rewards use per-cell combined objective | `_find_best` on `combined_objective` |
| `arm_elimination`, `epsilon_lucb`, `threshold` | Elimination / LUCB stats on combined per-sample objectives | same |
| `hill_climbing`, `bayesian` | Move / surrogate target uses combined objective | same |
| `brute_force`, `random` | Does not steer *which* combos to try | same |
| `lm_proposal` | Proposer uses `objective=` **text**, not these lambdas | `combined_objective` on the one evaluated combo only |

After `select_best()`, a final pass recomputes every result’s `combined_objective` against the **full-run** normalizer so rankings are comparable.

!!! note "`lm_proposal` vs lambdas"
    `LMProposalModelSelector(objective="...")` is a natural-language hint to the **proposer LLM**. It is separate from `lambda_cost` / `lambda_latency`, which only affect the scalar reward used for ranking and bandit methods.

## `select_best()`

```python
results = selector.select_best(
    parallel=False,        # If True, evaluate combos concurrently with asyncio
    max_concurrent=20,     # Total concurrent API-call budget across all combos
)
```

Returns a [`SelectionResults`](results.md). `parallel=True` requires `agent.run` to be either async or threadsafe; the selector splits `max_concurrent` between outer (combos) and inner (datapoints) loops based on dataset size.

!!! note "Automatic cleanup"
    `select_best()` calls `tracker.stop()` on return or exception — caches flush to disk, masters tear down. The tracker remains queryable; only `close()` (which `select_best` does not call) drops the remote backend's HTTP client.

## Choosing a method

| `method` | Algorithm | When to use |
|:---|:---|:---|
| `"auto"` (default) | Arm elimination | Strong best-arm identification at lower search cost than brute force. Same impl as `"arm_elimination"`. |
| `"brute_force"` | Evaluate every combo on the full dataset | Small search space; ground-truth comparison. |
| `"random"` | Random search | Cheap baseline. |
| `"hill_climbing"` | Greedy per-node | Large combinatorial spaces with weak coupling between nodes. |
| `"arm_elimination"` | Successive elimination | Best-arm identification with PAC-style guarantees. |
| `"epsilon_lucb"` | LUCB with tolerance | Stop once a combo is within ε of the best. |
| `"matrix_ucb"` / `"matrix_ucb_lrf"` | UCB exploiting cross-combo structure | Large model x datapoint matrices; `lrf` adds low-rank factorization. |
| `"threshold"` | Threshold bandit successive elimination | "Find all combos above accuracy θ" rather than the single best. |
| `"lm_proposal"` | LM-guided | Uses `node_descriptions` to propose combinations. |
| `"bayesian"` | Bayesian optimization | Optional extra: `pip install "agentopt-py[bayesian]"`. |

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

::: agentopt.model_selection.epsilon_lucb.EpsilonLUCBModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.matrix_ucb.MatrixUCBModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.matrix_ucb.MatrixUCBLRFModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.threshold_successive_elimination.ThresholdBanditSEModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.lm_proposal.LMProposalModelSelector
    options:
      members: false
      show_bases: false

::: agentopt.model_selection.bayesian_optimization.BayesianOptimizationModelSelector
    options:
      members: false
      show_bases: false
