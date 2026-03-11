"""LangChain-specific LLM builder, agent sync, and FrameworkAdapter."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..adapter import FrameworkAdapter, register_adapter
from ..constants import detect_provider, no_key_error

# Module prefixes that identify LangChain-compatible LLM objects.
LANGCHAIN_COMPATIBLE_PREFIXES = (
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google",
    "langchain_aws",
    "langchain_community",
)


def is_langchain_compatible_llm(llm: Any) -> bool:
    """Check if an LLM object is from a LangChain-compatible package."""
    return (type(llm).__module__ or "").startswith(LANGCHAIN_COMPATIBLE_PREFIXES)


# ---------------------------------------------------------------------------
# LangChain LLM factory
# ---------------------------------------------------------------------------


def _openrouter_fallback(model_name: str) -> Any:
    """Create a ChatOpenAI pointed at OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )


def _litellm_fallback(model_name: str) -> Any:
    """Create a ChatOpenAI pointed at a LiteLLM proxy server."""
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_API_BASE")
    if not api_key or not base_url:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )


def _proxy_fallback(model_name: str) -> Any:
    """Try LiteLLM first, then OpenRouter."""
    return _litellm_fallback(model_name) or _openrouter_fallback(model_name)


_PROVIDER_ENV = {
    "bedrock": "AWS_ACCESS_KEY_ID + AWS_DEFAULT_REGION",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "default": "OPENAI_API_KEY",
}


def _create_native_llm(provider: str, clean_name: str, api_key: str) -> Any:
    """Create a native LangChain LLM for the given provider."""
    if provider == "bedrock":
        aws_region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION"))
        if not aws_region:
            return None
        try:
            from langchain_aws import ChatBedrockConverse

            return ChatBedrockConverse(model=clean_name, region_name=aws_region)
        except ImportError:
            return None

    if provider == "openai" or provider == "default":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=clean_name, api_key=api_key)

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=clean_name, google_api_key=api_key)
        except ImportError:
            return None

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=clean_name, api_key=api_key)
        except ImportError:
            return None

    return None


def create_model_from_string(model_name: str) -> Any:
    """Create a LangChain model object from a string name.

    Automatically detects provider based on prefix/name and falls back
    to LiteLLM proxy → OpenRouter when native API keys are unavailable.

    Args:
        model_name: Model name string (e.g., "openai/gpt-4o", "claude-sonnet-4-20250514")

    Returns:
        LangChain model object
    """
    # --- LiteLLM explicit prefix (not handled by detect_provider) ---
    if model_name.startswith("litellm/"):
        clean_name = model_name.removeprefix("litellm/")
        fallback = _litellm_fallback(clean_name)
        if fallback:
            return fallback
        raise ValueError(
            f"LiteLLM requested for '{model_name}' but LITELLM_API_KEY and "
            "LITELLM_API_BASE are not both set."
        )

    provider, clean_name, api_key = detect_provider(model_name)

    if api_key:
        llm = _create_native_llm(provider, clean_name, api_key)
        if llm is not None:
            return llm

    fallback = _proxy_fallback(model_name)
    if fallback:
        return fallback
    raise no_key_error(_PROVIDER_ENV.get(provider, "OPENAI_API_KEY"), model_name)


def build_langchain_compatible_llm(model_name: str) -> Optional[Any]:
    """Create a LangChain-compatible chat model from a model name string.

    Returns a new model instance, or ``None`` on failure.
    """
    try:
        return create_model_from_string(model_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
except ImportError:
    _BaseCallbackHandler = object  # type: ignore[misc, assignment]


class _TokenCountingCallback(_BaseCallbackHandler):
    """LangChain callback handler that accumulates token usage."""

    def __init__(self) -> None:
        if _BaseCallbackHandler is not object:
            super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            self.input_tokens += usage.get("prompt_tokens", 0)
            self.output_tokens += usage.get("completion_tokens", 0)
            return
        # Fallback: per-generation usage_metadata (newer LangChain / Anthropic)
        for gens in getattr(response, "generations", []):
            for gen in gens:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                meta = getattr(msg, "usage_metadata", None) or {}
                self.input_tokens += meta.get("input_tokens", 0)
                self.output_tokens += meta.get("output_tokens", 0)

    def reset(self) -> Tuple[int, int]:
        in_tok, out_tok = self.input_tokens, self.output_tokens
        self.input_tokens = 0
        self.output_tokens = 0
        return in_tok, out_tok


class LangChainAdapter(FrameworkAdapter):
    """Adapter for LangChain ``AgentExecutor`` agents."""

    invoke_method_name = "invoke"

    def __init__(self) -> None:
        # Maps agent id → callback handler (installed on first get_invoke_fn call).
        self._callbacks: Dict[int, _TokenCountingCallback] = {}

    def get_token_usage(self, agent: Any) -> Tuple[int, int]:
        handler = self._callbacks.get(id(agent))
        if handler is None:
            return (0, 0)
        return handler.reset()

    def _extract_prompt(self, agent_executor: Any) -> Any:
        """Extract the ChatPromptTemplate from a LangChain AgentExecutor's chain."""
        chain = agent_executor.agent
        if chain is None:
            return None

        # Unwrap RunnableAgent / RunnableMultiActionAgent → inner RunnableSequence.
        if hasattr(chain, "runnable"):
            chain = chain.runnable

        # Prefer .steps (canonical RunnableSequence attribute) then fall back to
        # the .first/.middle/.last split for older LangChain versions.
        if hasattr(chain, "steps"):
            for item in chain.steps:
                if type(item).__name__ == "ChatPromptTemplate":
                    return item

        for attr in ("first", "middle", "last"):
            if not hasattr(chain, attr):
                continue
            obj = getattr(chain, attr)
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    if type(item).__name__ == "ChatPromptTemplate":
                        return item
            elif type(obj).__name__ == "ChatPromptTemplate":
                return obj

        return None

    @staticmethod
    def _sync_executor(executor: Any, llm: Any, tools: List[Any], prompt: Any) -> None:
        """Rebuild the LCEL agent chain inside an AgentExecutor with a new LLM."""
        try:
            from langchain_classic.agents import (
                AgentExecutor,
                create_tool_calling_agent,
            )
        except ImportError:
            from langchain.agents import AgentExecutor, create_tool_calling_agent

        new_agent = create_tool_calling_agent(llm, tools, prompt)
        temp = AgentExecutor(agent=new_agent, tools=tools)
        executor.agent = temp.agent

    def detect(self, agent: Any) -> bool:
        return (
            (type(agent).__module__ or "").startswith("langchain")
            and hasattr(agent, "agent")
            and hasattr(agent, "tools")
        )

    def detect_model(self, model: Any) -> bool:
        return is_langchain_compatible_llm(model)

    def _install_callback(self, agent: Any) -> None:
        """Attach a _TokenCountingCallback to the executor's callbacks list."""
        if id(agent) in self._callbacks:
            return
        handler = _TokenCountingCallback()
        self._callbacks[id(agent)] = handler
        agent.callbacks = list(agent.callbacks or []) + [handler]

    def get_invoke_fn(self, agent: Any) -> Callable:
        return agent.invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register token counting and (for agent= path) LCEL chain rebuild sync.

        Token tracking is installed on the LLM object itself for both paths —
        executor-level callbacks are not reliably propagated to on_llm_end in
        newer LangChain (0.3+) with LCEL-based AgentExecutor.
        """
        # Always install token tracking on the LLM itself.
        if id(proxy) not in self._callbacks:
            handler = _TokenCountingCallback()
            self._callbacks[id(proxy)] = handler

            def _reinstall(new_llm: Any, _h: _TokenCountingCallback = handler) -> None:
                new_llm.callbacks = list(getattr(new_llm, "callbacks", None) or []) + [
                    _h
                ]

            underlying = object.__getattribute__(proxy, "_optmodel")
            _reinstall(underlying)
            proxy._add_sync(_reinstall)

        if agent is None:
            return  # invoke_fn path: done

        # agent= path: also install LCEL chain rebuild sync.
        tools = agent.tools
        prompt = self._extract_prompt(agent)
        if prompt is None:
            return  # can't rebuild without the prompt template

        def _sync(
            new_llm: Any,
            _exec: Any = agent,
            _tools: list = tools,
            _prompt: Any = prompt,
        ) -> None:
            LangChainAdapter._sync_executor(_exec, new_llm, _tools, _prompt)

        proxy._add_sync(_sync)

    def wrap_invoke_fn_for_parallel(self, invoke_fn: Any) -> Any:
        """Wrap invoke_fn to inject a token-counting callback via RunnableConfig.

        The returned token_tracker is a _TokenCountingCallback whose counts
        are accessible via get_token_usage(token_tracker).
        LangGraph propagates RunnableConfig callbacks to all child runnables,
        so on_llm_end fires on the fresh LLMs created by clone_fn.
        """
        from langchain_core.runnables import RunnableConfig

        handler = _TokenCountingCallback()
        self._callbacks[id(handler)] = handler
        config = RunnableConfig(callbacks=[handler])

        def wrapped(input_data):
            return invoke_fn(input_data, config=config)

        return handler, wrapped

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
            from langchain_classic.agents import (
                AgentExecutor,
                create_tool_calling_agent,
            )
        except ImportError:
            from langchain.agents import AgentExecutor, create_tool_calling_agent

        tools = agent.tools
        prompt = self._extract_prompt(agent)
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
        fresh_llm = build_langchain_compatible_llm(model_name)
        if fresh_llm is None:
            raise RuntimeError(
                f"LangChainAdapter.clone_for_parallel: could not build LLM for '{model_name}'."
            )

        new_agent_chain = create_tool_calling_agent(fresh_llm, tools, prompt)
        temp = AgentExecutor(agent=new_agent_chain, tools=tools)
        # Shallow-copy the executor structure, replacing only the agent chain.
        return agent.model_copy(update={"agent": temp.agent}, deep=False)


# Self-register — runs when this module is first imported.
register_adapter(LangChainAdapter())
