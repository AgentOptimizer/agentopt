"""
Model proxy package: transparent LLM wrapping and framework-aware utilities.
"""

from .base import ModelProxy
from .utils import (
    MODEL_FIELDS,
    clone_agent,
    copy_with_replacements,
    create_variant,
    discover_proxy_targets,
    get_model_field_value,
    get_model_name,
    make_invoke_fn,
    resolve_model_name,
    set_model_field,
)

__all__ = [
    "ModelProxy",
    "MODEL_FIELDS",
    "clone_agent",
    "copy_with_replacements",
    "create_variant",
    "discover_proxy_targets",
    "get_model_field_value",
    "get_model_name",
    "make_invoke_fn",
    "resolve_model_name",
    "set_model_field",
]
