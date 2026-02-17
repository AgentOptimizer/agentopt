"""Unified LLM builder: dispatches to framework-specific builders."""

from typing import Any, Optional

from .constants import is_crewai_llm, is_langchain_llm
from .crewai import build_crewai_llm
from .langchain import build_langchain_llm


def build_llm(model_name: str, current_llm: Any) -> Optional[Any]:
    """Construct a fresh LLM via the originating framework's factory.

    Inspects ``type(current_llm).__module__`` to detect the framework,
    then delegates to the appropriate builder.

    Uses only the model name — no settings are carried over.  Each
    provider's factory resolves its own API key from the environment
    and applies its own defaults, avoiding parameter mismatches
    across models (e.g. ``max_tokens`` vs ``max_completion_tokens``).

    Returns a new LLM object, or ``None`` if the framework is not
    recognized or construction fails.
    """
    if is_crewai_llm(current_llm):
        return build_crewai_llm(model_name)

    if is_langchain_llm(current_llm):
        return build_langchain_llm(model_name)

    return None
