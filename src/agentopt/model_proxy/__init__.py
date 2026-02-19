"""Model proxy package: transparent LLM wrapper with model swapping."""

from .base import ModelProxy

try:
    from .openai_sdk import register_openai_agents_model

    register_openai_agents_model(ModelProxy)
except Exception:
    # Swallow any optional-dep issues (e.g., OpenAI Agents SDK pulling old TF).
    pass

__all__ = ["ModelProxy"]
