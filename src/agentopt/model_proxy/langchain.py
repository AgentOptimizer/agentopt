"""LangChain-specific LLM builder, agent sync, and FrameworkAdapter."""

from typing import Any, Callable, List, Optional


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


def extract_prompt(agent_executor: Any) -> Any:
    """Extract the ChatPromptTemplate from a LangChain AgentExecutor's chain.

    Previously lived in ``model_selection/utils.py``; moved here to avoid a
    circular import (``model_selection`` imports from ``model_proxy``, not the
    other way around).
    """
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


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class LangChainAdapter:
    """Adapter for LangChain ``AgentExecutor`` agents."""

    invoke_method_name = "invoke"

    def detect(self, agent: Any) -> bool:
        from .constants import is_langchain_executor

        return is_langchain_executor(agent)

    def get_invoke_fn(self, agent: Any) -> Callable:
        return agent.invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register a closure that rebuilds the LCEL chain on every model swap.

        Extracts tools and prompt automatically from the executor — users
        do not need to pass them manually.
        """
        tools = agent.tools
        prompt = extract_prompt(agent)
        if prompt is None:
            return  # can't rebuild without the prompt template

        def _sync(
            new_llm: Any,
            _exec: Any = agent,
            _tools: list = tools,
            _prompt: Any = prompt,
        ) -> None:
            sync_langchain_executor(_exec, new_llm, _tools, _prompt)

        proxy._add_sync(_sync)

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Rebuild a fresh AgentExecutor with a new LLM for this combination.

        Unlike CrewAI/LlamaIndex, LangChain bakes the LLM into an immutable
        LCEL chain at construction time, so a simple model_copy is not enough
        — we must rebuild the chain from scratch with a fresh LLM.

        This also fixes the previously broken LangChain parallel path where
        ``proxy_to_attr`` came back empty (LLM is nested inside the chain,
        not a direct attribute of the executor).
        """
        try:
            from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
        except ImportError:
            from langchain.agents import AgentExecutor, create_tool_calling_agent

        tools = agent.tools
        prompt = extract_prompt(agent)
        if prompt is None:
            raise RuntimeError(
                "LangChainAdapter.clone_for_parallel: no prompt found in executor — "
                "cannot rebuild chain."
            )

        # Single-proxy case (standard): build one fresh LLM for this combo.
        model_spec = combo[0]
        model_name = (
            model_spec if isinstance(model_spec, str) else get_model_name(model_spec)
        )
        fresh_llm = build_langchain_llm(model_name)
        if fresh_llm is None:
            raise RuntimeError(
                f"LangChainAdapter.clone_for_parallel: could not build LLM for '{model_name}'."
            )

        new_agent_chain = create_tool_calling_agent(fresh_llm, tools, prompt)
        temp = AgentExecutor(agent=new_agent_chain, tools=tools)
        # Shallow-copy the executor structure, replacing only the agent chain.
        return agent.model_copy(update={"agent": temp.agent}, deep=False)


# Self-register — runs when this module is first imported.
from .adapter import register_adapter  # noqa: E402

register_adapter(LangChainAdapter())
