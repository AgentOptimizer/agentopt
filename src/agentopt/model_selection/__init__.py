"""Model selection algorithms."""

from .arm_elimination import ArmEliminationModelSelector
from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults
from .brute_force import BruteForceModelSelector
from .matrix_ucb import MatrixUCBLRFModelSelector, MatrixUCBModelSelector

# Bayesian is optional (requires torch/botorch)
try:
    from .bayesian_optimization import BayesianOptimizationModelSelector
except ImportError:
    pass

__all__ = [
    "BaseModelSelector",
    "BruteForceModelSelector",
    "ArmEliminationModelSelector",
    "MatrixUCBModelSelector",
    "MatrixUCBLRFModelSelector",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
]
