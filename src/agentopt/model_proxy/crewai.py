"""
CrewAI-specific variant creation via the LLM factory.
"""

from typing import Any


def create_variant(original_llm: Any, model_name: str) -> Any:
    """Create a CrewAI LLM variant using the ``LLM(model=...)`` factory.

    This runs ``__init__`` on the resulting object, which properly
    initialises the HTTP client, sets cached flags, and strips
    provider prefixes (e.g. ``"openai/gpt-4o-mini"`` -> ``"gpt-4o-mini"``).
    """
    from crewai import LLM

    return LLM(model=model_name)
