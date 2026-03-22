"""
agentopt — Framework-agnostic LLM model selection optimizer for agents.

Uses the factory pattern: define an ``agent_fn(models: Dict[str, Any])``
that builds your agent with the given model candidates (string names or
LLM instances), then let a ModelSelector find the best combination.
"""

__version__ = "0.1.0"

from agentopt.proxy import CallRecord, LLMTracker

from .base_models import AgentFn, Dataset, EvalFn, ModelsConfig
from .model_selection import (
    ArmEliminationModelSelector,
    BaseModelSelector,
    BruteForceModelSelector,
    EpsilonLUCBModelSelector,
    HillClimbingModelSelector,
    LMProposalModelSelector,
    DatapointResult,
    ModelResult,
    RandomSearchModelSelector,
    SelectionResults,
    ThresholdBanditSEModelSelector,
)

# Default ModelSelector = BruteForceModelSelector
ModelSelector = BruteForceModelSelector

# Bayesian is optional (requires torch/botorch)
try:
    from .model_selection import BayesianOptimizationModelSelector
except ImportError:
    pass

__all__ = [
    # Metadata
    "__version__",
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
    "EpsilonLUCBModelSelector",
    "ThresholdBanditSEModelSelector",
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
