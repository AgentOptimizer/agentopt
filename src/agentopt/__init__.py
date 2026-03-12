"""
Model selection and optimization for LLM-powered agents using `ModelProxy`.

This package provides:
- `ModelProxy` as a transparent proxy that wraps LLM objects and allows model swapping
- model selection utilities (`BruteForceModelSelector`, `RandomSearchModelSelector`, `HillClimbingModelSelector`, `ArmEliminationModelSelector`, `HyperbandModelSelector`, `BayesianOptimizationModelSelector`) to choose among models
"""

from .model_proxy import ModelProxy
from .model_selection import (
    BruteForceModelSelector,
    RandomSearchModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
    HyperbandModelSelector,
    BayesianOptimizationModelSelector,
)
from .base_models import (
    Dataset,
    EvalFn,
    ModelSpec,
    ModelsConfig,
)
from .model_selection.base import BaseModelSelector, ModelResult, SelectionResults

# Backwards-compatible alias: default brute-force selector
ModelSelector = BruteForceModelSelector

__all__ = [
    # Core
    "ModelProxy",
    "BaseModelSelector",
    "BruteForceModelSelector",
    "RandomSearchModelSelector",
    "HillClimbingModelSelector",
    "ArmEliminationModelSelector",
    "HyperbandModelSelector",
    "BayesianOptimizationModelSelector",
    "ModelSelector",
    # Types
    "ModelResult",
    "SelectionResults",
    "Dataset",
    "EvalFn",
    "ModelSpec",
    "ModelsConfig",
]
