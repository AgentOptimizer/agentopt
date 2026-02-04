"""
Model selection and orchestration for LLM-style models using `ModelProxy` and invokers.

This package provides:
- `ModelProxy` as a unified interface over concrete model backends
- invoker implementations (e.g., LangChain, CrewAI) to actually call models
- model selection utilities (`ModelSelector`, `BaseModelSelector`) to choose among models
"""

from .model_proxy import ModelProxy
from .load_dataset import load_dataset
from .model_factory import create_model_from_string, normalize_models
from .model_selection import ModelSelector
from .types import (
    Message,
    EvaluationTask,
    AccuracyFn,
    ModelSpec,
    ModelsConfig,
)
from .invoker.base import InvokerProtocol
from .model_selection.base import BaseModelSelector, ModelResult, SelectionResults

__all__ = [
    # Core
    "ModelProxy",
    "BaseModelSelector",
    "ModelSelector",
    "load_dataset",  # [TODO: Temporary]
    "create_model_from_string",
    "normalize_models",
    # Types
    "InvokerProtocol",
    "Message",
    "EvaluationTask",
    "ModelResult",
    "SelectionResults",
    "AccuracyFn",
    "ModelSpec",
    "ModelsConfig",
]

# Optional LangChain adapter (only available if langchain is installed)
try:
    from .invoker.langchain import LangchainInvoker, ChainedLangchainInvoker

    __all__.append("LangchainInvoker")
    __all__.append("ChainedLangchainInvoker")
except ImportError:
    pass

# Optional CrewAI adapter
try:
    from .invoker.crewai import CrewInvoker

    __all__.append("CrewInvoker")
except ImportError:
    pass
