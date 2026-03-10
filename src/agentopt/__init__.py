"""
Model selection and optimization for LLM-powered agents using `ModelProxy`.

This package provides:
- `ModelProxy` as a transparent proxy that wraps LLM objects and allows model swapping
- model selection utilities (`BruteForceModelSelector`, `HillClimbingModelSelector`, `ArmEliminationModelSelector`, `BayesianOptimizationModelSelector`) to choose among models
"""

from .model_proxy import ModelProxy
from .model_proxy.framework_specific_implementation.langchain_compat import (
    create_model_from_string,
)
from .model_selection import (
    BruteForceModelSelector,
    RandomSearchModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
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
    "BayesianOptimizationModelSelector",
    "ModelSelector",
    "create_model_from_string",
    # Types
    "ModelResult",
    "SelectionResults",
    "Dataset",
    "EvalFn",
    "ModelSpec",
    "ModelsConfig",
]
