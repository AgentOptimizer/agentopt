"""Model selection algorithms."""

from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults
from .matrix_ucb import MatrixUCBModelSelector
from .gittins import GittinsModelSelector, GittinsRecommendation, GittinsSelectionResults
from .cohv_ucb import CoHVUCBModelSelector
from .random_search import RandomSearchModelSelector, RandomSelectionResults

# Bayesian is optional (requires torch/botorch)
try:
    from .bayesian_optimization import BayesianOptimizationModelSelector
except ImportError:
    pass

__all__ = [
    "BaseModelSelector",
    "MatrixUCBModelSelector",
    "CoHVUCBModelSelector",
    "RandomSearchModelSelector",
    "RandomSelectionResults",
    "GittinsModelSelector",
    "GittinsRecommendation",
    "GittinsSelectionResults",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
]
