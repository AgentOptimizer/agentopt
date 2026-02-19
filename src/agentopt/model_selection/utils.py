"""Utility helpers for model selection.

``extract_prompt`` has moved to ``model_proxy.langchain`` to avoid a circular
import (``model_selection`` imports from ``model_proxy``, not the reverse).
This re-export keeps any existing call-sites working.
"""

from ..model_proxy.langchain import extract_prompt  # noqa: F401

__all__ = ["extract_prompt"]
