"""AG2 (AutoGen 2) support for ModelProxy.

AG2 validates llm_config and rejects ModelProxy, so we patch
ConversableAgent._validate_llm_config to accept ModelProxy and
silently convert LLMConfig → AG2ConfigWrapper in ModelProxy.__init__
so that set_model() can update the config in place.

This is the same registration pattern used for OpenAI Agents SDK
in openai_sdk.py.
"""

from __future__ import annotations

import os
from typing import Any


class AG2ConfigWrapper:
    """Mutable wrapper around AG2's config_list.

    Exposes a ``model`` property so ModelProxy.set_model() can find and
    update the model name.  The config_list is shared by reference with
    the LLMConfig created during validation, so in-place mutations
    propagate to the agent automatically.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.config_list = [
            {
                "api_type": "openai",
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ]

    @property
    def model(self) -> str:
        return self.config_list[0]["model"]

    @model.setter
    def model(self, value: str) -> None:
        self.config_list[0]["model"] = value


def _is_ag2_llm_config(obj: Any) -> bool:
    """Check if an object is an AG2 LLMConfig."""
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith("autogen") and type(obj).__name__ == "LLMConfig"


def register_ag2_llm_config(proxy_cls: type) -> None:
    """Register *proxy_cls* to work transparently with AG2's LLMConfig.

    1. Patches ``ModelProxy.__init__`` to silently convert LLMConfig →
       AG2ConfigWrapper so ``set_model()`` works via the wrapper's
       ``model`` property.
    2. Patches ``ConversableAgent._validate_llm_config`` to accept
       ModelProxy and return a linked LLMConfig whose config_list is
       shared by reference with the wrapper.

    Called at import time from ``model_proxy/__init__.py``.
    """
    from autogen import ConversableAgent, LLMConfig

    # --- Patch ModelProxy.__init__ ---
    original_init = proxy_cls.__init__

    def _patched_init(self: Any, initial_model: Any) -> None:
        if _is_ag2_llm_config(initial_model):
            # Extract model name from the LLMConfig
            config_list = list(initial_model.config_list)
            if config_list:
                first = config_list[0]
                cfg = first if isinstance(first, dict) else first.model_dump()
                model_name = cfg.get("model", "gpt-4o-mini")
            else:
                model_name = "gpt-4o-mini"
            wrapper = AG2ConfigWrapper(model_name)
            original_init(self, wrapper)
        else:
            original_init(self, initial_model)

    proxy_cls.__init__ = _patched_init

    # --- Patch ConversableAgent._validate_llm_config ---
    original_validate = ConversableAgent._validate_llm_config.__func__

    @classmethod  # type: ignore[misc]
    def _patched_validate(cls: type, llm_config: Any) -> Any:
        if isinstance(llm_config, proxy_cls):
            wrapper = object.__getattribute__(llm_config, "_optmodel")
            if isinstance(wrapper, AG2ConfigWrapper):
                # Create LLMConfig that shares config_list with wrapper
                return LLMConfig(config_list=wrapper.config_list)
        return original_validate(cls, llm_config)

    ConversableAgent._validate_llm_config = _patched_validate


def extract_ag2_content(response: Any) -> str:
    """Extract text content from an AG2 agent response.

    AG2's agent.run() returns a response object whose events must be consumed
    before the summary/messages are available.
    """
    for _ in response.events:
        pass
    if hasattr(response, "summary") and response.summary:
        return response.summary
    if response.messages:
        last_msg = response.messages[-1]
        content = getattr(last_msg, "content", None) or str(last_msg)
        return content
    return ""
