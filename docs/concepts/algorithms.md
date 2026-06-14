# Selection Algorithms

AgentOpt provides 5 selection algorithms. Choose based on your search space size and evaluation budget.

## At a Glance

| Algorithm | Strategy | Evaluations | Best For |
|:----------|:---------|:------------|:---------|
| [Brute Force](#brute-force) | Exhaustive | All | Small spaces (< 50 combos) |
| [Arm Elimination](#arm-elimination) | Progressive pruning | Adaptive | Statistical early stopping |
| [Matrix UCB](#matrix-ucb) | UCB over combo × datapoint grid | Budgeted | Large spaces with selective datapoint sampling |
| [Bayesian Optimization](#bayesian-optimization) | GP surrogate | Sequential | Expensive evaluations |

!!! tip "Common interface"
    All selectors share the same constructor and `select_best()` method. Switching algorithms is a one-line change.

    ```python
    selector = AnySelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
    )
    results = selector.select_best(parallel=True, max_concurrent=20)
    ```

    To optionally weight cost and latency against accuracy, pass `lambda_cost` and/or `lambda_latency` (default `0.0`). See [Combined objective](../api/selectors.md#combined-objective-optional-costlatency-weights).

---

## Brute Force

Evaluates every combination in the Cartesian product.

```python
from agentopt import BruteForceModelSelector

selector = BruteForceModelSelector(
    agent=MyAgent,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
)
```

!!! success "When to use"
    Small search spaces where you can afford to evaluate everything. Guarantees finding the true optimum.

!!! warning "Complexity"
    Evaluations grow as the product of model list sizes. 5 models x 3 nodes = 125 combinations.

---

## Arm Elimination

Progressively eliminates statistically dominated combinations. Starts with a small batch of datapoints, then grows the batch while eliminating underperformers.

```python
from agentopt import ArmEliminationModelSelector

selector = ArmEliminationModelSelector(
    agent=MyAgent,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    growth_factor=2.0,
    confidence=1.0,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `n_initial` | `None` | Initial batch size. Default: 10% of dataset (`max(1, len(dataset)//10)`) |
| `growth_factor` | `2.0` | Batch size multiplier per round |
| `confidence` | `1.0` | Elimination confidence threshold |

!!! success "When to use"
    When bad combinations should be eliminated early to save budget. Particularly effective when there are clearly weak options. This is the default (`method="auto"`).

---

## Matrix UCB

UCB exploration over the combination × datapoint matrix. Instead of evaluating every combo on every datapoint, it adaptively picks which cells to observe next.

```python
from agentopt import MatrixUCBModelSelector

selector = MatrixUCBModelSelector(
    agent=MyAgent,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    a=1.0,
    sample_fraction=0.25,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `a` | `1.0` | UCB exploration coefficient |
| `sample_fraction` | `None` | Fraction of the combo × datapoint grid to observe (alias for `observation_budget_fraction`) |
| `seed` | `None` | Random seed for reproducibility |

A low-rank factorization variant is available via `MatrixUCBLRFModelSelector` (`method="matrix_ucb_lrf"`). It adds parameters like `rank`, `ensemble_size`, and `warmup_fraction` for structured uncertainty over the matrix.

!!! success "When to use"
    Large search spaces where you want to sample both combinations and datapoints intelligently rather than running the full grid.

---

## Bayesian Optimization

Uses a Gaussian Process surrogate to predict accuracy for unevaluated combinations, then selects the most promising one via Expected Improvement.

```python
from agentopt import BayesianOptimizationModelSelector

selector = BayesianOptimizationModelSelector(
    agent=MyAgent,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    batch_size=1,
    sample_fraction=0.25,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `batch_size` | `1` | Combinations to evaluate per GP iteration |
| `sample_fraction` | `0.25` | Fraction of dataset to use per evaluation |

!!! note "Extra dependency"
    Requires PyTorch and BoTorch:
    ```bash
    pip install "agentopt-py[bayesian]"
    ```

!!! success "When to use"
    When each evaluation is expensive (large dataset, slow models) and you want to minimize total evaluations. The GP learns from past results to pick the most informative next combination.
