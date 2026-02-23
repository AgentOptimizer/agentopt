"""
AG2 (AutoGen 2) support for ModelProxy.

AG2 validates llm_config and rejects ModelProxy, so we use a mutable config
wrapper that ModelProxy updates in place. No agent registration or sync is
needed — invoke_fn captures the proxy and agents use LLMConfig(config_list=wrapper.config_list).
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Tuple

from .base import ModelProxy

# Optional: autogen is required for AG2
try:
    from autogen import ConversableAgent, LLMConfig
except ImportError as e:
    raise ImportError(
        "AG2 support requires the autogen package. Install with: pip install autogen"
    ) from e


class AG2ConfigWrapper:
    """
    Mutable wrapper for AG2 LLMConfig. AG2 validates llm_config and rejects
    ModelProxy, so we use a real LLMConfig backed by mutable config_list.
    ModelProxy wraps this wrapper and updates config_list[0]['model'] when
    set_model() is called.
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


def _run_agent_and_get_content(agent: ConversableAgent, message: str) -> str:
    """Run agent.run() and return the last message content or summary."""
    response = agent.run(
        message=message,
        max_turns=2,
        user_input=False,
    )
    for _ in response.events:
        pass
    if hasattr(response, "summary") and response.summary:
        return response.summary
    if response.messages:
        last_msg = response.messages[-1]
        content = getattr(last_msg, "content", None) or str(last_msg)
        return content
    return ""


def create_single_agent_proxy_and_invoke(
    default_model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful math assistant. Answer questions concisely with the final numerical answer.",
) -> Tuple[ModelProxy, Callable[[Any], str]]:
    """
    Create a single AG2 agent wired for model selection.

    Returns:
        (proxy, invoke_fn). Use invoke_fn(input_data) with input_data["input"] as the user message.
        BruteForceModelSelector will call proxy.set_model() before each invoke.
    """
    wrapper = AG2ConfigWrapper(default_model)
    llm_config = LLMConfig(config_list=wrapper.config_list)
    proxy = ModelProxy(wrapper)

    agent = ConversableAgent(
        name="math_assistant",
        system_message=system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    def invoke_fn(input_data: Any) -> str:
        return _run_agent_and_get_content(agent, input_data["input"])

    return proxy, invoke_fn


def create_multi_agent_proxies_and_invoke(
    default_model: str = "gpt-4o-mini",
    researcher_system_message: str = "You are a research assistant. Find and summarize information.",
    coder_system_message: str = "You are a coding assistant. Write efficient code based on research.",
) -> Tuple[Tuple[ModelProxy, ModelProxy], Callable[[Any], str]]:
    """
    Create a two-agent AG2 setup (researcher + coder) with separate proxies for model selection.

    Returns:
        ((researcher_proxy, coder_proxy), invoke_fn). BruteForceModelSelector can optimize
        each proxy independently.
    """
    config_research = AG2ConfigWrapper(default_model)
    config_coder = AG2ConfigWrapper(default_model)
    researcher_proxy = ModelProxy(config_research)
    coder_proxy = ModelProxy(config_coder)

    researcher = ConversableAgent(
        name="researcher",
        system_message=researcher_system_message,
        llm_config=LLMConfig(config_list=config_research.config_list),
        human_input_mode="NEVER",
    )

    coder = ConversableAgent(
        name="coder",
        system_message=coder_system_message,
        llm_config=LLMConfig(config_list=config_coder.config_list),
        human_input_mode="NEVER",
    )

    def invoke_fn(input_data: Any) -> str:
        question = input_data["input"]
        research_output = _run_agent_and_get_content(researcher, question)
        return _run_agent_and_get_content(
            coder, f"Context: {research_output}\n\nTask: {question}"
        )

    return (researcher_proxy, coder_proxy), invoke_fn
