"""
Framework adapters for agentopt.
"""

from .base import InvokerProtocol

__all__ = ["InvokerProtocol"]

try:
    from .langchain import LangchainInvoker, ChainedLangchainInvoker

    __all__.extend(["LangchainInvoker", "ChainedLangchainInvoker"])
except ImportError:
    # LangChain not installed
    pass

try:
    from .crewai import CrewInvoker

    __all__.append("CrewInvoker")
except ImportError:
    # CrewAI not installed
    pass
