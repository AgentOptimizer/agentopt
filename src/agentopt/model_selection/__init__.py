"""Model selection algorithms."""

from .arm_elimination import ArmEliminationModelSelector
from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults
from .brute_force import BruteForceModelSelector
from .epsilon_lucb import EpsilonLUCBModelSelector
from .hill_climbing import HillClimbingModelSelector
from .lm_proposal import LMProposalModelSelector
from .random_search import RandomSearchModelSelector
from .threshold_successive_elimination import ThresholdBanditSEModelSelector
from .matrix_ucb import MatrixUCBLRFModelSelector, MatrixUCBModelSelector
from .objectives import ObjectiveMode

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
    "MatrixUCBModelSelector",
    "MatrixUCBLRFModelSelector",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
    "ObjectiveMode",
]
