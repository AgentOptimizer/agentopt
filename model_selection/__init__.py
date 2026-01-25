"""
Model selection package - bind models to agents with explicit attribute names.
"""
from .core import bind_model, ModelProxy, ModelSelector, load_dataset
from .model_factory import create_model_from_string, normalize_models

__all__ = [
    "bind_model",
    "ModelProxy",
    "ModelSelector",
    "load_dataset",  # [TODO: Temporary]
    "create_model_from_string",
    "normalize_models",
]
