"""
Model factory for creating LangChain model objects from string names.
Handles provider detection and fallback to OpenRouter if API keys are missing.
"""
from typing import Any, Union, List, Dict
import os
from dotenv import load_dotenv

load_dotenv()


def create_model_from_string(model_name: str) -> Any:
    """
    Create a LangChain model object from a string name.
    
    Automatically detects provider based on prefix:
    - "openai/" → OpenAI (if OPENAI_API_KEY exists, else OpenRouter)
    - "google/" or "gemini" → Google Gemini (if GOOGLE_API_KEY exists, else OpenRouter)
    - "anthropic/" or "claude" → Anthropic (if ANTHROPIC_API_KEY exists, else OpenRouter)
    - Otherwise → OpenRouter
    
    Args:
        model_name: Model name string (e.g., "openai/gpt-4o", "google/gemini-3-flash-preview")
    
    Returns:
        LangChain model object
    """
    # Check for OpenRouter fallback
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Determine provider from model name
    if model_name.startswith("openai/"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            # Use OpenAI directly
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name.replace("openai/", ""),
                api_key=openai_api_key,
            )
        elif openrouter_api_key:
            # Fallback to OpenRouter
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                base_url=openrouter_base_url,
                api_key=openrouter_api_key,
            )
        else:
            raise ValueError(
                "Neither OPENAI_API_KEY nor OPENROUTER_API_KEY found. "
                "Please set one of them in your .env file."
            )
    
    elif model_name.startswith("google/") or "gemini" in model_name.lower():
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            # Use Google directly
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Extract model name (remove "google/" prefix if present)
                clean_name = model_name.replace("google/", "")
                return ChatGoogleGenerativeAI(
                    model=clean_name,
                    google_api_key=google_api_key,
                )
            except ImportError:
                # Fallback to OpenRouter if langchain_google_genai not available
                if openrouter_api_key:
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(
                        model=model_name,
                        base_url=openrouter_base_url,
                        api_key=openrouter_api_key,
                    )
                else:
                    raise ValueError(
                        "langchain_google_genai not installed and OPENROUTER_API_KEY not found. "
                        "Please install langchain-google-genai or set OPENROUTER_API_KEY."
                    )
        elif openrouter_api_key:
            # Fallback to OpenRouter
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                base_url=openrouter_base_url,
                api_key=openrouter_api_key,
            )
        else:
            raise ValueError(
                "Neither GOOGLE_API_KEY nor OPENROUTER_API_KEY found. "
                "Please set one of them in your .env file."
            )
    
    elif model_name.startswith("anthropic/") or "claude" in model_name.lower():
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            # Use Anthropic directly
            try:
                from langchain_anthropic import ChatAnthropic
                # Extract model name (remove "anthropic/" prefix if present)
                clean_name = model_name.replace("anthropic/", "")
                return ChatAnthropic(
                    model=clean_name,
                    api_key=anthropic_api_key,
                )
            except ImportError:
                # Fallback to OpenRouter if langchain_anthropic not available
                if openrouter_api_key:
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(
                        model=model_name,
                        base_url=openrouter_base_url,
                        api_key=openrouter_api_key,
                    )
                else:
                    raise ValueError(
                        "langchain_anthropic not installed and OPENROUTER_API_KEY not found. "
                        "Please install langchain-anthropic or set OPENROUTER_API_KEY."
                    )
        elif openrouter_api_key:
            # Fallback to OpenRouter
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                base_url=openrouter_base_url,
                api_key=openrouter_api_key,
            )
        else:
            raise ValueError(
                "Neither ANTHROPIC_API_KEY nor OPENROUTER_API_KEY found. "
                "Please set one of them in your .env file."
            )
    
    else:
        # Default to OpenRouter for unknown formats
        if openrouter_api_key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                base_url=openrouter_base_url,
                api_key=openrouter_api_key,
            )
        else:
            raise ValueError(
                f"Unknown model format '{model_name}' and OPENROUTER_API_KEY not found. "
                "Please set OPENROUTER_API_KEY in your .env file or use a recognized format."
            )


def normalize_models(models: Union[Dict[str, List[Union[str, Any]]], Dict[str, List[str]], Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
    """
    Normalize models input: convert string model names to model objects.
    
    Args:
        models: Dictionary mapping attribute paths to list of model names (strings) or model objects
    
    Returns:
        Dictionary mapping attribute paths to list of model objects
    """
    normalized = {}
    for attr_path, model_list in models.items():
        normalized_list = []
        for model in model_list:
            if isinstance(model, str):
                # Convert string to model object
                normalized_list.append(create_model_from_string(model))
            else:
                # Already a model object
                normalized_list.append(model)
        normalized[attr_path] = normalized_list
    return normalized
