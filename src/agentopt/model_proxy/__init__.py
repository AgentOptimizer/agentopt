"""Model proxy package: transparent LLM wrapper with model swapping."""

from .base import ModelProxy

try:
    from .openai_sdk import register_openai_agents_model

    register_openai_agents_model(ModelProxy)
except Exception:
    # Swallow any optional-dep issues (e.g., OpenAI Agents SDK pulling old TF).
    pass

try:
    from .ag2 import register_ag2_llm_config

    register_ag2_llm_config(ModelProxy)
except Exception:
    # Swallow any optional-dep issues (autogen not installed).
    pass

__all__ = ["ModelProxy"]
