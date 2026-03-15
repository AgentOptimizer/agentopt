"""Helpers for safe model candidate copying."""

import copy
from typing import Any

from pydantic import BaseModel


def clone_model_spec(model_spec: Any) -> Any:
    """Return an isolated copy for thread-safe parallel evaluation."""
    if isinstance(model_spec, BaseModel):
        try:
            return model_spec.model_copy(deep=True)
        except Exception:
            pass

    try:
        return copy.deepcopy(model_spec)
    except Exception:
        try:
            return copy.copy(model_spec)
        except Exception:
            return model_spec
