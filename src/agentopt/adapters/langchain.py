"""
LangChain wrapper for model selection compatibility.

Provides OptLangchainAgent wrapper that is compatible with ModelSelector's interface.
"""

from typing import Any, Dict, List, Optional
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from agentopt import ModelProxy
from agentopt.types import AgentProtocol

load_dotenv()


class OptLangchainAgent(AgentProtocol):
    """
    LangChain agent wrapper compatible with ModelSelector.

    Usage:
        from langchain_core.tools import tool

        @tool
        def add(a: float, b: float) -> float:
            '''Add two numbers.'''
            return a + b

        agent = OptLangchainAgent(
            model="openai/gpt-4o-mini",
            tools=[add],
            system_prompt="You are a helpful math assistant.",
        )

        selector = ModelSelector(
            agent=agent,
            models={"._model": ["openai/gpt-4o-mini", "openai/gpt-4o"]},
            accuracy_fn=check_accuracy,
        )
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        tools: Optional[List[Any]] = None,
        system_prompt: str = "You are a helpful assistant.",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Create a LangChain agent with ModelProxy for model selection.

        Args:
            model: Model name (OpenRouter format, e.g., "openai/gpt-4o")
            tools: List of LangChain tools
            system_prompt: System prompt for the agent
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            base_url: Base URL (defaults to OPENROUTER_BASE_URL or OpenRouter)
        """
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )

        if not self._api_key:
            raise ValueError(
                "API key not found. Please set OPENROUTER_API_KEY in your .env file "
                "or pass api_key parameter."
            )

        # Create initial model
        initial_model = ChatOpenAI(
            model=model,
            base_url=self._base_url,
            api_key=self._api_key,
        )

        # Create proxy for model swapping
        self._model = ModelProxy("model", initial_model)

        # Create agent with proxy model
        self._agent = create_agent(
            model=self._model,
            tools=tools or [],
            system_prompt=system_prompt,
        )

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent with the given input.

        Args:
            input_dict: Dictionary with "messages" key containing list of messages

        Returns:
            Dictionary with "messages" key containing response
        """
        return self._agent.invoke(input_dict)
