"""Model selection algorithms."""

from .arm_elimination import ArmEliminationModelSelector
from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults
from .brute_force import BruteForceModelSelector
from .epsilon_lucb import EpsilonLUCBModelSelector
from .hill_climbing import HillClimbingModelSelector
from .lm_proposal import LMProposalModelSelector
from .random_search import RandomSearchModelSelector
from .streaming_brute_force import StreamingBruteForceModelSelector
from .streaming_random_search import StreamingRandomSearchModelSelector
from .threshold_successive_elimination import ThresholdBanditSEModelSelector

# Bayesian is optional (requires torch/botorch)
try:
    from .bayesian_optimization import BayesianOptimizationModelSelector
except ImportError:
    pass

__all__ = [
    "BaseModelSelector",
    "BruteForceModelSelector",
    "RandomSearchModelSelector",
    "HillClimbingModelSelector",
    "ArmEliminationModelSelector",
    "EpsilonLUCBModelSelector",
    "ThresholdBanditSEModelSelector",
    "LMProposalModelSelector",
    "StreamingBruteForceModelSelector",
    "StreamingRandomSearchModelSelector",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
]
