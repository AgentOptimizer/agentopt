"""Model selection algorithms."""

from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults
from .matrix_ucb import MatrixUCBModelSelector
from .gittins import GittinsModelSelector, GittinsRecommendation, GittinsSelectionResults
from .random_search import RandomSearchModelSelector, RandomSelectionResults
from .cost_aware import CostAwareUCBModelSelector, CostAwareQNEHVIModelSelector, CostAwareSelectionResults

# Bayesian is optional (requires torch/botorch)
try:
    from .bayesian_optimization import BayesianOptimizationModelSelector
except ImportError:
    pass

__all__ = [
    "BaseModelSelector",
    "MatrixUCBModelSelector",
    "RandomSearchModelSelector",
    "RandomSelectionResults",
    "CostAwareUCBModelSelector",
    "CostAwareQNEHVIModelSelector",
    "CostAwareSelectionResults",
    "GittinsModelSelector",
    "GittinsRecommendation",
    "GittinsSelectionResults",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
]
