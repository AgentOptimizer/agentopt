"""Utility helpers for model selection."""

from typing import Any


def extract_prompt(agent_executor: Any) -> Any:
    """Extract the ChatPromptTemplate from a LangChain AgentExecutor's chain."""
    chain = getattr(agent_executor, "agent", None)
    if chain is None:
        return None
    for attr in ("first", "middle", "last"):
        obj = getattr(chain, attr, None)
        if obj is None:
            continue
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if type(item).__name__ == "ChatPromptTemplate":
                    return item
        elif type(obj).__name__ == "ChatPromptTemplate":
            return obj
    return None
