"""OpenAI Agents SDK compatibility: ABC registration and model builder."""

from typing import Any

from .constants import MODEL_FIELDS


def build_openai_agents_model(model_name: str) -> Any:
    """Create an OpenAI Agents SDK ``Model`` from a model name string.

    Uses the default ``OpenAIProvider`` to resolve the name (e.g.
    ``"gpt-4o-mini"``) into a concrete ``Model`` instance.

    Returns the ``Model``.
    """
    from agents.models.openai_provider import OpenAIProvider

    provider = OpenAIProvider()
    return provider.get_model(model_name)


def _get_model_name(proxy: Any) -> str:
    """Extract the current model name string from the proxy's wrapped object."""
    model = object.__getattribute__(proxy, "_optmodel")
    for field in MODEL_FIELDS:
        if hasattr(model, field):
            return str(getattr(model, field))
    raise ValueError("Cannot determine model name from wrapped object")


async def _get_response(self: Any, *args: Any, **kwargs: Any) -> Any:
    """OpenAI Agents SDK ``Model`` interface — non-streaming.

    Delegates to the wrapped model if it implements ``get_response``,
    otherwise resolves a real SDK ``Model`` from the current model name.
    """
    model = object.__getattribute__(self, "_optmodel")
    if hasattr(model, "get_response"):
        return await model.get_response(*args, **kwargs)
    resolved = build_openai_agents_model(_get_model_name(self))
    return await resolved.get_response(*args, **kwargs)


def _stream_response(self: Any, *args: Any, **kwargs: Any) -> Any:
    """OpenAI Agents SDK ``Model`` interface — streaming variant."""
    model = object.__getattribute__(self, "_optmodel")
    if hasattr(model, "stream_response"):
        return model.stream_response(*args, **kwargs)
    resolved = build_openai_agents_model(_get_model_name(self))
    return resolved.stream_response(*args, **kwargs)


def register_openai_agents_model(proxy_cls: type) -> None:
    """Register *proxy_cls* as a virtual subclass of the OpenAI Agents SDK
    ``Model`` ABC and patch the required interface methods onto it.

    1. ``Model.register(proxy_cls)`` — makes ``isinstance(proxy, Model)``
       return ``True``.
    2. Attaches ``get_response`` and ``stream_response`` so the SDK can
       invoke them at runtime.
    """
    from agents.models.interface import Model

    Model.register(proxy_cls)
    proxy_cls.get_response = _get_response
    proxy_cls.stream_response = _stream_response
