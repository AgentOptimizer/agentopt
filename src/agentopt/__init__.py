"""
Model selection and optimization for LLM-powered agents using `ModelProxy`.

This package provides:
- `ModelProxy` as a transparent proxy that wraps LLM objects and allows model swapping
- model selection utilities (`ModelSelector`, `BaseModelSelector`) to choose among models
"""

from .model_proxy import ModelProxy
from .model_factory import create_model_from_string, normalize_models
from .model_selection import ModelSelector
from .base_models import (
    EvalFn,
    ModelSpec,
    ModelsConfig,
)
from .model_selection.base import BaseModelSelector, ModelResult, SelectionResults
from .optimize import optimize, parallel_select

__all__ = [
    # Core
    "ModelProxy",
    "BaseModelSelector",
    "ModelSelector",
    "create_model_from_string",
    "normalize_models",
    "optimize",
    "parallel_select",
    # Types
    "ModelResult",
    "SelectionResults",
    "EvalFn",
    "ModelSpec",
    "ModelsConfig",
]
