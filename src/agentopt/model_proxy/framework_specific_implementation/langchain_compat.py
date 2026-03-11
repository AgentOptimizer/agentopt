"""LangChain-specific LLM builder, agent sync, and FrameworkAdapter."""

import os
from typing import Any, Callable, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration

from ..adapter import FrameworkAdapter
from ..constants import detect_provider, no_key_error


class _TokenTrackingCallback(BaseCallbackHandler):
    """LangChain callback that feeds on_llm_end token counts into a TokenAccumulator."""

    def __init__(self, tracker: Any) -> None:
        super().__init__()
        self._tracker = tracker

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                if isinstance(gen, ChatGeneration):
                    meta = getattr(gen.message, "usage_metadata", None)
                    if meta:
                        self._tracker.add(
                            meta.get("input_tokens", 0),
                            meta.get("output_tokens", 0),
                        )


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


class LangChainAdapter(FrameworkAdapter):
    """Adapter for LangChain ``AgentExecutor`` agents."""

    invoke_method_name = "invoke"

    def __init__(self) -> None:
        # Maps id(agent_copy) → TokenAccumulator for parallel clones.
        self._clone_trackers: dict = {}

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

    def get_invoke_fn(self, agent: Any) -> Callable:
        callback = getattr(self, "_pending_callback", None)
        if callback is not None:
            original_invoke = agent.invoke

            def _invoke_with_tracking(input_data: Any) -> Any:
                return original_invoke(input_data, config={"callbacks": [callback]})

            return _invoke_with_tracking
        return agent.invoke

    def create_token_tracker(self, agent: Any = None) -> Any:
        """Return the tracker pre-attached to a parallel clone's proxy, if any."""
        if agent is not None:
            tracker = self._clone_trackers.pop(id(agent), None)
            if tracker is not None:
                self._pending_callback = _TokenTrackingCallback(tracker)
                return tracker
        from ..token_tracking import TokenAccumulator

        tracker = TokenAccumulator()
        self._pending_callback = _TokenTrackingCallback(tracker)
        return tracker

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register LCEL chain rebuild sync and token tracking callback."""
        if agent is None:
            return  # invoke_fn path: no agent to attach to

        # Install LCEL chain rebuild sync.
        tools = agent.tools
        prompt = self._extract_prompt(agent)
        if prompt is not None:

            def _sync(
                new_llm: Any,
                _exec: Any = agent,
                _tools: list = tools,
                _prompt: Any = prompt,
                _proxy: Any = proxy,
            ) -> None:
                LangChainAdapter._sync_executor(_exec, _proxy, _tools, _prompt)

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

        # Wrap in a per-thread proxy so token interception works in parallel.
        from ..proxy import ModelProxy
        from ..token_tracking import TokenAccumulator

        fresh_proxy = ModelProxy(fresh_llm)
        tracker = TokenAccumulator()
        fresh_proxy._set_token_tracker(tracker)

        new_agent_chain = create_tool_calling_agent(fresh_proxy, tools, prompt)
        temp = AgentExecutor(agent=new_agent_chain, tools=tools)
        # Shallow-copy the executor structure, replacing only the agent chain.
        clone = agent.model_copy(update={"agent": temp.agent}, deep=False)
        self._clone_trackers[id(clone)] = tracker
        return clone


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

# Self-register — runs when this module is first imported.
register_adapter(LangChainAdapter())
