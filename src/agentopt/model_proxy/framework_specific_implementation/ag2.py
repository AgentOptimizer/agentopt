"""AG2 (AutoGen 2) support for ModelProxy.

AG2 validates llm_config and rejects ModelProxy, so we patch
ConversableAgent._validate_llm_config to accept ModelProxy and
silently convert LLMConfig → AG2ConfigWrapper in ModelProxy.__init__
so that set_model() can update the config in place.

Additionally, ConversableAgent.__init__ is patched to auto-register
agents with the proxy, so _sync_registered_frameworks() can recreate
the LLMConfig and force client recreation on set_model().

This is the same registration pattern used for OpenAI Agents SDK
in openai_sdk.py.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Callable, List


def is_ag2_agent(agent: Any) -> bool:
    """Check if an agent is an AG2 ConversableAgent."""
    module = getattr(type(agent), "__module__", "") or ""
    return module.startswith("autogen") and hasattr(agent, "run")


def is_ag2_llm(llm: Any) -> bool:
    """Check if an LLM object is an AG2ConfigWrapper."""
    return isinstance(llm, AG2ConfigWrapper)


def _build_ag2_config(model: str) -> dict:
    """Build an AG2 config_list entry with correct api_type and key for the model."""
    bare = model.split("/", 1)[-1] if "/" in model else model
    if bare.startswith("claude") or model.startswith("anthropic/"):
        return {
            "api_type": "anthropic",
            "model": bare,
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        }
    # Default to OpenAI-compatible
    return {
        "api_type": "openai",
        "model": bare,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }


class AG2ConfigWrapper:
    """Mutable wrapper around AG2's config_list.

    Exposes a ``model`` property so ModelProxy.set_model() can find and
    update the model name.  The config_list is shared by reference with
    the LLMConfig created during validation, so in-place mutations
    propagate to the agent automatically.
    """

    def __init__(self, model: str) -> None:
        self.config_list = [_build_ag2_config(model)]

    @property
    def model(self) -> str:
        return self.config_list[0]["model"]

    @model.setter
    def model(self, value: str) -> None:
        self.config_list[0] = _build_ag2_config(value)


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
    3. Patches ``ConversableAgent.__init__`` to auto-register agents
       with the proxy for explicit sync on ``set_model()``.

    Called at import time from ``model_proxy/__init__.py``.
    """
    from autogen import ConversableAgent, LLMConfig

    # --- Patch ModelProxy.__init__ ---
    original_init = proxy_cls.__init__

    def _patched_init(self: Any, initial_model: Any) -> None:
        if _is_ag2_llm_config(initial_model):
            # Extract model name from the LLMConfig
            config_list = list(initial_model.config_list)
            if not config_list:
                raise ValueError("LLMConfig has an empty config_list")
            first = config_list[0]
            cfg = first if isinstance(first, dict) else first.model_dump()
            model_name = cfg.get("model")
            if not model_name:
                raise ValueError("LLMConfig entry missing 'model' key")
            wrapper = AG2ConfigWrapper(model_name)
            original_init(self, wrapper)
            # Initialize _ag2_agents list so the patched ConversableAgent.__init__
            # can append agents to it.
            object.__setattr__(self, "_ag2_agents", [])
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
                return LLMConfig(*wrapper.config_list)
        return original_validate(cls, llm_config)

    ConversableAgent._validate_llm_config = _patched_validate

    # --- Patch ConversableAgent.__init__ for auto-registration ---
    original_agent_init = ConversableAgent.__init__

    def _patched_agent_init(self_agent: Any, *args: Any, **kwargs: Any) -> None:
        llm_config_arg = kwargs.get("llm_config")
        original_agent_init(self_agent, *args, **kwargs)
        # Auto-register agent with the proxy so _sync_registered_frameworks
        # can recreate LLMConfig + force client recreation on set_model().
        if isinstance(llm_config_arg, proxy_cls):
            ag2_agents = object.__getattribute__(llm_config_arg, "_ag2_agents")
            if self_agent not in ag2_agents:
                ag2_agents.append(self_agent)

    ConversableAgent.__init__ = _patched_agent_init


def sync_ag2_agents(agents: List[Any], wrapper: Any) -> None:
    """Recreate LLMConfig for registered AG2 agents from the updated wrapper.

    Called by ``ModelProxy._sync_registered_frameworks()`` after
    ``set_model()`` has already mutated the wrapper's config_list.
    This ensures AG2 agents pick up the change even if they cache
    an ``OpenAIWrapper`` client at init time.
    """
    from autogen import LLMConfig

    if not isinstance(wrapper, AG2ConfigWrapper):
        return
    new_config = LLMConfig(*wrapper.config_list)
    for agent in agents:
        agent.llm_config = new_config
        # Force client recreation if AG2 caches it
        # Recreate client from new config (setting to None silently disables LLM)
        agent.client = agent._create_client(new_config)


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


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class AG2Adapter:
    """Adapter for AG2 ``ConversableAgent`` objects."""

    invoke_method_name = "run"

    def detect(self, agent: Any) -> bool:
        return is_ag2_agent(agent)

    def get_invoke_fn(self, agent: Any) -> Callable:
        """Wrap agent.run() to handle input dict and extract content."""

        def _invoke(input_data: Any) -> Any:
            if isinstance(input_data, dict):
                message = input_data.get("input", str(input_data))
            else:
                message = str(input_data)
            response = agent.run(message=message, max_turns=1, user_input=False)
            return extract_ag2_content(response)

        return _invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register sync callback that calls sync_ag2_agents on set_model()."""
        try:
            ag2_agents = object.__getattribute__(proxy, "_ag2_agents")
        except AttributeError:
            ag2_agents: list = []
            object.__setattr__(proxy, "_ag2_agents", ag2_agents)

        if agent not in ag2_agents:
            ag2_agents.append(agent)

        def _sync(new_llm: Any, _agents: list = ag2_agents) -> None:
            if isinstance(new_llm, AG2ConfigWrapper):
                sync_ag2_agents(_agents, new_llm)

        proxy._add_sync(_sync)

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Clone the AG2 agent with a fresh LLMConfig for this combination."""
        from autogen import LLMConfig

        model_spec = combo[0]
        model_name = (
            model_spec if isinstance(model_spec, str) else get_model_name(model_spec)
        )
        new_config = LLMConfig(_build_ag2_config(model_name))
        agent_copy = copy.copy(agent)
        agent_copy.llm_config = new_config
        agent_copy.client = agent_copy._create_client(new_config)
        return agent_copy


# Self-register — runs when this module is first imported.
try:
    from ..adapter import register_adapter  # noqa: E402

    register_adapter(AG2Adapter())
except Exception:
    pass
