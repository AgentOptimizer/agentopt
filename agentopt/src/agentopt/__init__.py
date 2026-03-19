"""
agentopt — Framework-agnostic LLM model selection optimizer for agents.

Uses the factory pattern: define an ``agent_fn(models: Dict[str, Any])``
that builds your agent with the given model candidates (string names or
LLM instances), then let a ModelSelector find the best combination.
"""

from agentproxy import CallRecord, LLMTracker

from .base_models import AgentFn, Dataset, EvalFn, ModelsConfig
from .model_selection import (
    ArmEliminationModelSelector,
    BaseModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    HyperbandModelSelector,
    LMProposalModelSelector,
    DatapointResult,
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
    "LLMTracker",
    "CallRecord",
    # Selectors
    "BruteForceModelSelector",
    "RandomSearchModelSelector",
    "HillClimbingModelSelector",
    "ArmEliminationModelSelector",
    "HyperbandModelSelector",
    "LMProposalModelSelector",
    "BayesianOptimizationModelSelector",
    # Result types
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
    # Type aliases
    "AgentFn",
    "Dataset",
    "EvalFn",
    "ModelsConfig",
]
