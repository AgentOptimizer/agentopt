"""
Custom agent function for LangChain.
"""
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from model_selection import ModelProxy
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b


def MyLangchainAgent(model_name: str = "openai/gpt-4o"):
    """
    Create a LangChain agent.
    User's original agent code goes here.
    
    Args:
        model_name: Name of the model to use (OpenRouter format)
    
    Returns:
        Agent object that can have its model bound externally
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Please set it in your .env file or environment variables."
        )
    
    # Create initial model
    initial_model = ChatOpenAI(
        model=model_name,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    ### original code
    # agent = create_agent(
    #     model=initial_model,
    #     tools=[get_weather],
    #     system_prompt="You are a helpful assistant",
    # )

    ### new code for model selection
    # Create a proxy model that will forward to the current model
    # This allows us to swap models after agent creation
    proxy_model = ModelProxy("model", initial_model)
    
    # Create agent with proxy model
    agent = create_agent(
        model=proxy_model,
        tools=[add],
        system_prompt="You are a helpful math assistant. Use the add tool when you need to add numbers.",
    )
    
    # Store proxy reference on agent for later access
    agent._model = proxy_model
    
    return agent
