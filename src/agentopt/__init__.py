"""
agentopt — Framework-agnostic LLM model selection optimizer for agents.

Uses the factory pattern: define an ``agent_fn(models: Dict[str, str])``
that builds your agent with the given model names, then let a ModelSelector
find the best combination.
"""

from .base_models import AgentFn, Dataset, EvalFn, ModelsConfig
from .model_selection import (
    ArmEliminationModelSelector,
    BaseModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    HyperbandModelSelector,
    ModelResult,
    RandomSearchModelSelector,
    SelectionResults,
)

# Default ModelSelector = BruteForceModelSelector
ModelSelector = BruteForceModelSelector

# Bayesian is optional (requires torch/botorch)
try:
    from .model_selection import BayesianOptimizationModelSelector
except ImportError:
    pass

__all__ = [
    # Core API
    "ModelSelector",
    "BaseModelSelector",
    # Selectors
    "BruteForceModelSelector",
    "RandomSearchModelSelector",
    "HillClimbingModelSelector",
    "ArmEliminationModelSelector",
    "HyperbandModelSelector",
    "BayesianOptimizationModelSelector",
    # Result types
    "ModelResult",
    "SelectionResults",
    # Type aliases
    "AgentFn",
    "Dataset",
    "EvalFn",
    "ModelsConfig",
]
