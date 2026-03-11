"""Unified LLM builder: dispatches to framework-specific builders."""

from typing import Any, Optional

from .framework_specific_implementation.ag2 import AG2ConfigWrapper, is_ag2_llm
from .framework_specific_implementation.crewai import build_crewai_llm, is_crewai_llm
from .framework_specific_implementation.langchain_compat import (
    build_langchain_compatible_llm,
    is_langchain_compatible_llm,
)
from .framework_specific_implementation.llamaindex import (
    is_llamaindex_llm,
    build_llamaindex_llm,
)
from .framework_specific_implementation.openai_sdk import (
    build_openai_agents_model,
)


def _is_openai_sdk_model(obj: Any) -> bool:
    """Return True if *obj* is an OpenAI Agents SDK Model or a SimpleNamespace placeholder."""
    mod = type(obj).__module__ or ""
    if mod.startswith("agents"):
        return True
    # SimpleNamespace with a 'model' attribute is the convention for the OpenAI SDK examples.
    if type(obj).__name__ == "SimpleNamespace" and hasattr(obj, "model"):
        return True
    return False


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

    if is_langchain_compatible_llm(current_llm):
        return build_langchain_compatible_llm(model_name)

    if is_llamaindex_llm(current_llm):
        return build_llamaindex_llm(model_name)

    if is_ag2_llm(current_llm):
        return AG2ConfigWrapper(model_name)

    if _is_openai_sdk_model(current_llm):
        return build_openai_agents_model(model_name)

    return None
