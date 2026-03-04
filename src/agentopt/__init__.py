"""
Model selection and optimization for LLM-powered agents using `ModelProxy`.

This package provides:
- `ModelProxy` as a transparent proxy that wraps LLM objects and allows model swapping
- model selection utilities (`ModelSelector`, `BaseModelSelector`) to choose among models
"""

from .model_proxy import ModelProxy
from .model_factory import create_model_from_string, normalize_models
from .model_selection import (
    BruteForceModelSelector,
    HillClimbingModelSelector,
    ArmEliminationModelSelector,
)
from .base_models import (
    Dataset,
    EvalFn,
    ModelSpec,
    ModelsConfig,
)
from .model_selection.base import BaseModelSelector, ModelResult, SelectionResults

__all__ = [
    # Core
    "ModelProxy",
    "BaseModelSelector",
    "BruteForceModelSelector",
    "HillClimbingModelSelector",
    "ArmEliminationModelSelector",
    "create_model_from_string",
    "normalize_models",
    # Types
    "ModelResult",
    "SelectionResults",
    "Dataset",
    "EvalFn",
    "ModelSpec",
    "ModelsConfig",
]
