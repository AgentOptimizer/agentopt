# Selection Strategies

AgentOpt provides six model selection strategies, from exhaustive to intelligent search.

## Brute Force (default)

Grid search over the full Cartesian product of all candidate models. Thorough but scales as O(n^k) where n is models per proxy and k is the number of proxies.

```python
from agentopt import BruteForceModelSelector

selector = BruteForceModelSelector(
    models={llm: ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]},
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
)
results = selector.select_best(parallel=True)
```

`ModelSelector` is an alias for `BruteForceModelSelector`.

## Random Search

Evaluates a random subset of the Cartesian product. Useful when brute force is too expensive.

```python
from agentopt import RandomSearchModelSelector

selector = RandomSearchModelSelector(
    models={llm: candidates},
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
    sample_fraction=0.25,  # evaluate 25% of combinations
)
results = selector.select_best(parallel=True)
```

## Hill Climbing

Local search using model quality/speed rankings. Starts from an initial combination and iteratively swaps one model at a time, keeping improvements. Much faster for large search spaces.

```python
from agentopt import HillClimbingModelSelector

selector = HillClimbingModelSelector(
    models={llm: candidates},
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
)
results = selector.select_best()
```

!!! warning "Experimental"
    Hill climbing may get stuck in local optima. It works best when model quality is roughly monotonic (better models generally score higher).

## Arm Elimination

Bandit-style successive elimination. Evaluates combinations in rounds with growing batch sizes and drops statistically dominated arms using confidence bounds. Often reduces total API calls versus brute force.

```python
from agentopt import ArmEliminationModelSelector

selector = ArmEliminationModelSelector(
    models={llm: candidates},
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
)
results = selector.select_best(parallel=True)
```

## Hyperband

Full Hyperband algorithm treating dataset samples as the resource. Runs multiple successive-halving brackets with different starting budgets.

```python
from agentopt import HyperbandModelSelector

selector = HyperbandModelSelector(
    models={llm: candidates},
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
    reduction_factor=3.0,  # eta parameter
)
results = selector.select_best(parallel=True)
```

The `reduction_factor` (eta) controls how aggressively candidates are eliminated. Higher values mean more aggressive pruning.

## Bayesian Optimization

Gaussian process-based optimization that models the accuracy surface and uses an acquisition function to select the most promising combination to evaluate next. Most efficient for large search spaces.

```python
from agentopt import BayesianOptimizationModelSelector

selector = BayesianOptimizationModelSelector(
    models={llm: candidates},
    eval_fn=eval_fn,
    dataset=dataset,
    agent=agent,
)
results = selector.select_best()
```

!!! note
    Requires the `bayesian` extra: `uv sync --extra bayesian`

## Comparison

| Strategy | Completeness | API Calls | Best for |
|----------|-------------|-----------|----------|
| Brute Force | All combinations | O(n^k) | Small search spaces, guaranteed optimal |
| Random Search | Sampled subset | Configurable | Medium spaces, budget-constrained |
| Hill Climbing | Local neighborhood | O(n*k) | Large spaces, when quality is monotonic |
| Arm Elimination | Adaptive | Varies | When many candidates are clearly weak |
| Hyperband | Multi-bracket | Varies | Large spaces with clear performance gaps |
| Bayesian | Guided search | Low | Very large spaces, expensive evaluations |
