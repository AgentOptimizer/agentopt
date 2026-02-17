"""LangChain-specific LLM builder and agent sync."""

from typing import Any, List, Optional


def build_langchain_llm(model_name: str) -> Optional[Any]:
    """Create a LangChain chat model from a model name string.

    Delegates to :func:`agentopt.model_factory.create_model_from_string`.
    Returns a new model instance, or ``None`` on failure.
    """
    try:
        from ..model_factory import create_model_from_string

        return create_model_from_string(model_name)
    except Exception:
        return None


def sync_langchain_executor(
    executor: Any, llm: Any, tools: List[Any], prompt: Any
) -> None:
    """Rebuild the LCEL agent chain inside an AgentExecutor with a new LLM."""
    try:
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:
        from langchain.agents import AgentExecutor, create_tool_calling_agent

    new_agent = create_tool_calling_agent(llm, tools, prompt)
    # AgentExecutor.__init__ wraps a RunnableSequence in an adapter that
    # provides .input_keys.  Assigning the raw sequence directly would skip
    # that wrapping, so we let a temporary executor do it for us.
    temp = AgentExecutor(agent=new_agent, tools=tools)
    executor.agent = temp.agent
