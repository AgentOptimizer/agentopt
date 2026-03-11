"""LlamaIndex-specific agent sync.

LlamaIndex FunctionAgents use strict Pydantic validation, so ModelProxy
cannot be passed as ``llm=`` directly. Instead, the agent is created with
a real LLM and ``register_llamaindex_agent()`` ensures that
``agent.llm`` is updated whenever ``proxy.set_model()`` is called.
"""

import logging
import os

from ..adapter import FrameworkAdapter
from ..token_tracking import TokenAccumulator
from typing import Any, Callable, List, Optional

from pydantic import BaseModel

from ..constants import detect_provider, no_key_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LlamaIndex LLM factory
# ---------------------------------------------------------------------------

# Provider → env var name (for error messages)
_PROVIDER_ENV = {
    "bedrock": "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_DEFAULT_REGION",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "default": "OPENAI_API_KEY",
}


def _openrouter_fallback(model_name: str) -> Any:
    """Route any model through OpenRouter via LlamaIndex's OpenRouter integration."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    from llama_index.llms.openrouter import OpenRouter

    # OpenRouter requires prefixed names for non-OpenAI models.
    if "/" not in model_name:
        lower = model_name.lower()
        if "claude" in lower:
            model_name = f"anthropic/{model_name}"
        elif "gemini" in lower:
            model_name = f"google/{model_name}"

    return OpenRouter(
        model=model_name,
        api_key=api_key,
        is_function_calling_model=True,
    )


def _create_native_llm(provider: str, clean_name: str, api_key: str) -> Any:
    """Create a native LlamaIndex LLM for the given provider."""
    if provider == "bedrock":
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION"))
        if not aws_secret or not aws_region:
            return None
        try:
            from llama_index.llms.anthropic import Anthropic

            return Anthropic(
                model=clean_name,
                aws_region=aws_region,
                aws_access_key_id=api_key,
                aws_secret_access_key=aws_secret,
            )
        except ImportError:
            return None

    if provider == "openai" or provider == "default":
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=clean_name, api_key=api_key)

    if provider == "google":
        try:
            from llama_index.llms.gemini import Gemini

            return Gemini(model=clean_name, api_key=api_key)
        except ImportError:
            return None

    if provider == "anthropic":
        try:
            from llama_index.llms.anthropic import Anthropic

            return Anthropic(model=clean_name, api_key=api_key)
        except ImportError:
            return None

    return None


def build_llamaindex_llm(model_name: str) -> Any:
    """Create a LlamaIndex LLM object from a model name string.

    Uses shared ``detect_provider()`` for provider detection, then maps
    to LlamaIndex-specific LLM classes. Falls back to OpenRouter.
    """
    provider, clean_name, api_key = detect_provider(model_name)

    if api_key:
        llm = _create_native_llm(provider, clean_name, api_key)
        if llm is not None:
            return llm

    fallback = _openrouter_fallback(model_name)
    if fallback:
        return fallback
    raise no_key_error(_PROVIDER_ENV.get(provider, "OPENAI_API_KEY"), model_name)


def is_llamaindex_llm(llm: Any) -> bool:
    """Check if an LLM object is from LlamaIndex."""
    return (type(llm).__module__ or "").startswith("llama_index")


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class LlamaIndexAdapter(FrameworkAdapter):
    """Adapter for LlamaIndex ``FunctionAgent`` and ``AgentWorkflow`` agents."""

    invoke_method_name = "run"

    def _install_token_handler(self, tracker: TokenAccumulator) -> Optional[Any]:
        """Attach a TokenCountingHandler via the global Settings callback manager.

        The handler feeds token counts into *tracker* after each LLM call.
        """
        try:
            from llama_index.core.callbacks import TokenCountingHandler
            from llama_index.core import Settings
            import tiktoken
        except ImportError:
            return None

        handler = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
        )
        Settings.callback_manager.add_handler(handler)
        return handler

    def create_token_tracker(self, agent: Any = None) -> TokenAccumulator:
        """Create a TokenAccumulator backed by LlamaIndex's TokenCountingHandler."""
        tracker = TokenAccumulator()
        handler = self._install_token_handler(tracker)
        if handler is not None:
            self._handler = handler
            self._tracker = tracker
        return tracker

    def detect(self, agent: Any) -> bool:
        return (type(agent).__module__ or "").startswith("llama_index") and hasattr(
            agent, "run"
        )

    def get_invoke_fn(self, agent: Any) -> Callable:
        """Wrap agent.run() so that the WorkflowHandler is awaited correctly.

        ``inspect.iscoroutinefunction(agent.run)`` returns False due to
        instrumentation decorators, so we use ``inspect.isawaitable()`` instead.

        Uses a persistent event loop to avoid "Event loop is closed" errors
        when LlamaIndex caches async resources (locks, connections) across
        multiple invocations of the same workflow.
        """
        import asyncio
        import inspect
        import threading

        handler = getattr(self, "_handler", None)
        tracker = getattr(self, "_tracker", None)

        method = agent.run

        # Create a dedicated event loop running in a background thread.
        # This loop stays open for the lifetime of the invoke_fn, so
        # LlamaIndex's cached async resources remain valid.
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(target=_loop.run_forever, daemon=True)
        _thread.start()

        def _invoke(input_data: Any) -> Any:
            async def _async_run() -> Any:
                if isinstance(input_data, dict):
                    result = method(**input_data)
                else:
                    result = method(input_data)
                if inspect.isawaitable(result):
                    return await result
                return result

            future = asyncio.run_coroutine_threadsafe(_async_run(), _loop)
            result = future.result()
            # Flush handler counts into the shared tracker.
            if handler is not None and tracker is not None:
                tracker.add(
                    handler.prompt_llm_token_count,
                    handler.completion_llm_token_count,
                )
                handler.reset_counts()
            return result

        return _invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register closures that push a new LLM into LlamaIndex agents."""
        is_workflow = hasattr(agent, "agents")  # AgentWorkflow has .agents dict

        if not is_workflow:
            # Simple FunctionAgent — direct llm assignment.
            def _sync(new_llm: Any, _agent: Any = agent) -> None:
                _agent.llm = new_llm

            proxy._add_sync(_sync)
            return

        # AgentWorkflow — positional or broadcast mapping.
        agents_dict = agent.agents
        agent_list = list(agents_dict.values())
        n_proxies = len(all_proxies)
        n_agents = len(agent_list)

        if n_proxies == 1:

            def _sync(
                new_llm: Any, _agents: List[Any] = list(agents_dict.values())
            ) -> None:
                for _ag in _agents:
                    _ag.llm = new_llm

            proxy._add_sync(_sync)

        elif n_proxies == n_agents:
            idx = all_proxies.index(proxy)
            sub_ag = agent_list[idx]

            def _sync(new_llm: Any, _ag: Any = sub_ag) -> None:
                _ag.llm = new_llm

            proxy._add_sync(_sync)

        else:
            logger.warning(
                "LlamaIndexAdapter: cannot map %d proxies to %d agents — "
                "sync not registered.",
                n_proxies,
                n_agents,
            )

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Create a completely fresh AgentWorkflow with cloned sub-agents.

        AgentWorkflow is NOT a Pydantic BaseModel — it's a Workflow with
        internal mutable state (Context, memory, step registry).
        ``model_copy(deep=False)`` doesn't work and ``copy.copy`` shares
        internal state across threads.  The only safe approach is to
        reconstruct the workflow from scratch with fresh agent clones.
        """
        if not hasattr(agent, "agents"):
            # Simple FunctionAgent — model_copy works fine
            cloned = (
                agent.model_copy(deep=False) if isinstance(agent, BaseModel) else agent
            )
            return cloned

        # AgentWorkflow — reconstruct with fresh cloned agents
        from llama_index.core.agent.workflow import AgentWorkflow

        agents_dict = agent.agents
        n_proxies = len(proxies)
        n_agents = len(agents_dict)
        cloned_agents = {}

        if n_proxies == 1:
            model_spec = combo[0]
            model_name = (
                model_spec
                if isinstance(model_spec, str)
                else get_model_name(model_spec)
            )
            fresh_llm = build_llamaindex_llm(model_name)
            for name, ag in agents_dict.items():
                cloned_agents[name] = ag.model_copy(
                    update={"llm": fresh_llm}, deep=False
                )
        elif n_proxies == n_agents:
            for (name, ag), model_spec in zip(agents_dict.items(), combo):
                model_name = (
                    model_spec
                    if isinstance(model_spec, str)
                    else get_model_name(model_spec)
                )
                fresh_llm = build_llamaindex_llm(model_name)
                cloned_agents[name] = ag.model_copy(
                    update={"llm": fresh_llm}, deep=False
                )
        else:
            logger.warning(
                "Cannot map %d proxies to %d agents for parallel clone.",
                n_proxies,
                n_agents,
            )
            return agent

        # Reconstruct a fresh AgentWorkflow — no shared state with original
        return AgentWorkflow(
            agents=list(cloned_agents.values()),
            root_agent=agent.root_agent,
        )


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

register_adapter(LlamaIndexAdapter())
