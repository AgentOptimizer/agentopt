"""
Model selection package - bind models to agents with explicit attribute names.
"""

from .core import bind_model, ModelProxy, ModelSelector, load_dataset
from .model_factory import create_model_from_string, normalize_models

__all__ = [
    "bind_model",
    "ModelProxy",
    "ModelSelector",
    "load_dataset",  # [TODO: Temporary]
    "create_model_from_string",
    "normalize_models",
]

# Optional LangChain adapter (only available if langchain is installed)
try:
    from .adapters.langchain import OptLangchainAgent

    __all__.append("OptLangchainAgent")
except ImportError:
    pass

# Optional CrewAI adapters (only available if crewai is installed)
try:
    from .adapters.crewai import OptCrewAgent, OptCrew

    __all__.extend(["OptCrewAgent", "OptCrew"])
except ImportError:
    pass
