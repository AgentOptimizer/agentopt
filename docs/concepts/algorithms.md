# Selection Algorithms

AgentOpt provides 7 selection algorithms. The right choice depends on your search space size and evaluation budget.

## Overview

| Algorithm | Evaluations | Best for |
|-----------|-------------|----------|
| [Brute Force](#brute-force) | All combinations | Small spaces (< 50 combos) |
| [Random Search](#random-search) | Random sample | Quick baseline |
| [Hill Climbing](#hill-climbing) | Guided neighbors | Medium spaces with model topology |
| [Arm Elimination](#arm-elimination) | Progressive pruning | Statistical early stopping |
| [Hyperband](#hyperband) | Multi-bracket halving | Large spaces, limited budget |
| [LM Proposal](#lm-proposal) | LLM-guided shortlist | Leveraging model knowledge |
| [Bayesian Optimization](#bayesian-optimization) | GP-guided search | Expensive evaluations |

All selectors share the same interface:

```python
selector = SomeSelector(
    agent_fn=agent_maker,    # factory: models dict → callable agent
    models=models,           # {node_name: [candidate_models]}
    eval_fn=eval_fn,         # (expected, actual) → float
    dataset=dataset,         # [(input, expected), ...]
)
results = selector.select_best(parallel=True, max_concurrent=20)
```

## Brute Force

Evaluates every combination in the Cartesian product. Simple and thorough.

```python
from agentopt import BruteForceModelSelector

selector = BruteForceModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
)
```

**When to use**: Small search spaces where you can afford to evaluate everything.

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

**When to use**: Quick exploration to get a baseline before committing to a more thorough search.

## Hill Climbing

Greedy search with random restarts. Uses model quality and speed rankings to define neighbors, so each iteration makes an informed single-step move.

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

**When to use**: Medium-sized spaces where you want to exploit model topology (cheaper models are neighbors of expensive ones).

## Arm Elimination

Progressively eliminates statistically dominated combinations. Starts with a small batch of datapoints, then grows the batch size while eliminating combinations that are statistically worse than others.

```python
from agentopt import ArmEliminationModelSelector

selector = ArmEliminationModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    n_initial=10,        # initial batch size
    growth_factor=2.0,   # batch size multiplier per round
    confidence=1.0,      # elimination confidence threshold
)
```

**When to use**: When you want early stopping — bad combinations are eliminated quickly, saving evaluation budget.

## Hyperband

Implements the full Hyperband algorithm using dataset samples as the resource. Runs multiple brackets of successive halving with different initial resource allocations.

```python
from agentopt import HyperbandModelSelector

selector = HyperbandModelSelector(
    agent_fn=agent_maker,
    models=models,
    eval_fn=eval_fn,
    dataset=dataset,
    reduction_factor=3.0,
)
```

**When to use**: Large search spaces with limited evaluation budget. Balances exploration (many combinations, few datapoints) with exploitation (few combinations, many datapoints).

## LM Proposal

Uses a proposer LLM to shortlist promising combinations before evaluation. The proposer sees the candidate models and dataset preview, then suggests which combinations to try.

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

**When to use**: When you want to leverage LLM knowledge about model capabilities to focus the search.

## Bayesian Optimization

Uses a Gaussian Process surrogate model to predict accuracy for unevaluated combinations, then selects the most promising one via Expected Improvement.

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

!!! note
    Requires optional dependencies: `pip install "agentopt[bayesian]"`

**When to use**: When each evaluation is expensive and you want to minimize the total number of evaluations.
