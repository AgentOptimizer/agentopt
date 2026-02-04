"""
Framework adapters for agentopt.
"""

__all__ = []

try:
    from .langchain import OptLangchainAgent

    __all__.append("OptLangchainAgent")
except ImportError:
    # LangChain not installed
    pass

try:
    from .crewai import OptCrewAgent, OptCrew

    __all__.extend(["OptCrewAgent", "OptCrew"])
except ImportError:
    # CrewAI not installed
    pass
