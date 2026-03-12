# API Reference

## Public exports

```python
from agentopt import (
    # Core
    ModelProxy,                      # Transparent LLM proxy
    ModelSelector,                   # Brute-force selector (default)
    ResponseCache,                   # File-backed response cache
    NoCache,                         # No-op cache drop-in

    # Selection strategies
    BruteForceModelSelector,         # Grid search (= ModelSelector)
    RandomSearchModelSelector,       # Random subset search
    HillClimbingModelSelector,       # Local search with restarts
    ArmEliminationModelSelector,     # Bandit-style elimination
    HyperbandModelSelector,          # Multi-bracket successive halving
    BayesianOptimizationModelSelector,  # GP-based optimization
    BaseModelSelector,               # Abstract base class

    # Results
    ModelResult,                     # Single model evaluation result
    SelectionResults,                # Container for all results

    # Type aliases
    EvalFn,                          # Callable[[str, Any], bool | float]
    Dataset,                         # Sequence[tuple[Any, Any]]
    ModelSpec,                       # str | Any
    ModelsConfig,                    # dict[ModelProxy, list[ModelSpec]]
)
```

## Modules

- [ModelProxy](model-proxy.md) — transparent proxy with model swapping
- [Model Selection](model-selection.md) — selectors, results, and base class
- [Cache](cache.md) — response caching for benchmarks
- [Types](types.md) — type aliases and dataset validation
