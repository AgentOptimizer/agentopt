"""Model proxy package: transparent LLM wrapper with model swapping."""

from .base import ModelProxy

try:
    from .openai_sdk import register_openai_agents_model

    register_openai_agents_model(ModelProxy)
except ImportError:
    pass

__all__ = ["ModelProxy"]
