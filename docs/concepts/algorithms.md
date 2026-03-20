# Selection Algorithms

AgentOpt provides 6 selection algorithms. Choose based on your search space size and evaluation budget.

## At a Glance

| Algorithm | Strategy | Evaluations | Best For |
|:----------|:---------|:------------|:---------|
| [Brute Force](#brute-force) | Exhaustive | All | Small spaces (< 50 combos) |
| [Random Search](#random-search) | Sampling | Configurable fraction | Quick baselines |
| [Hill Climbing](#hill-climbing) | Greedy + restarts | Guided neighbors | Medium spaces |
| [Arm Elimination](#arm-elimination) | Progressive pruning | Adaptive | Statistical early stopping |
| [LM Proposal](#lm-proposal) | LLM-guided | Shortlist | Leveraging model knowledge |
| [Bayesian Optimization](#bayesian-optimization) | GP surrogate | Sequential | Expensive evaluations |

!!! tip "Common interface"
    All selectors share the same constructor and `select_best()` method. Switching algorithms is a one-line change.

    ```python
    selector = AnySelector(
        agent_fn=agent_maker,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
    )
    results = selector.select_best(parallel=True, max_concurrent=20)
    ```

---

## Brute Force

Evaluates every combination in the Cartesian product.

```python
from agentopt import BruteForceModelSelector

selector = BruteForceModelSelector(
    agent_fn=agent_maker,
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

## Random Search

Samples a random fraction of all combinations.

```python
from agentopt import RandomSearchModelSelector

selector = RandomSearchModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    sample_fraction=0.25,  # evaluate 25% of combinations
    seed=42,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `sample_fraction` | `0.25` | Fraction of combinations to evaluate |
| `seed` | `None` | Random seed for reproducibility |

!!! success "When to use"
    Quick exploration to establish a baseline before committing to a thorough search.

---

## Hill Climbing

Greedy local search with random restarts. Defines "neighbors" using model quality and speed rankings, so each step is an informed single-model swap.

```python
from agentopt import HillClimbingModelSelector

selector = HillClimbingModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    max_iterations=20,
    num_restarts=3,
    patience=3,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `max_iterations` | `20` | Max steps per restart |
| `num_restarts` | `3` | Number of random restarts |
| `patience` | `3` | Steps without improvement before restart |

!!! success "When to use"
    Medium-sized spaces where you want to exploit model topology — cheaper models are neighbors of expensive ones.

---

## Arm Elimination

Progressively eliminates statistically dominated combinations. Starts with a small batch of datapoints, then grows the batch while eliminating underperformers.

```python
from agentopt import ArmEliminationModelSelector

selector = ArmEliminationModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    n_initial=10,
    growth_factor=2.0,
    confidence=1.0,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `n_initial` | `10` | Initial batch size (datapoints) |
| `growth_factor` | `2.0` | Batch size multiplier per round |
| `confidence` | `1.0` | Elimination confidence threshold |

!!! success "When to use"
    When bad combinations should be eliminated early to save budget. Particularly effective when there are clearly weak options.

---

## LM Proposal

Uses a proposer LLM to shortlist promising combinations before evaluation. The proposer sees the candidate models and a dataset preview, then suggests which combinations to try.

```python
from agentopt import LMProposalModelSelector

selector = LMProposalModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    proposer_model="gpt-4o-mini",
    max_combinations=12,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `proposer_model` | `"gpt-4o-mini"` | Model used for proposal generation |
| `max_combinations` | `12` | Max combinations to shortlist |

!!! success "When to use"
    When you want to leverage an LLM's knowledge about model capabilities to skip obviously bad combinations.

---

## Bayesian Optimization

Uses a Gaussian Process surrogate to predict accuracy for unevaluated combinations, then selects the most promising one via Expected Improvement.

```python
from agentopt import BayesianOptimizationModelSelector

selector = BayesianOptimizationModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    n_initial_random=5,
    n_iterations=20,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `n_initial_random` | `5` | Random combinations to seed the GP |
| `n_iterations` | `20` | GP-guided iterations after seeding |

!!! note "Extra dependency"
    Requires PyTorch and BoTorch:
    ```bash
    pip install "agentopt[bayesian]"
    ```

!!! success "When to use"
    When each evaluation is expensive (large dataset, slow models) and you want to minimize total evaluations. The GP learns from past results to pick the most informative next combination.
