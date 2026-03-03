"""LlamaIndex-specific agent sync.

LlamaIndex FunctionAgents use strict Pydantic validation, so ModelProxy
cannot be passed as ``llm=`` directly. Instead, the agent is created with
a real LLM and ``register_llamaindex_agent()`` ensures that
``agent.llm`` is updated whenever ``proxy.set_model()`` is called.
"""

import logging
import os
from typing import Any, Callable, List

from pydantic import BaseModel

from ..constants import MODEL_FIELDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LlamaIndex LLM factory — provider detection + API key fallback
# ---------------------------------------------------------------------------


def _openrouter_model_name(model_name: str) -> str:
    """Ensure model name has the provider prefix OpenRouter expects.

    OpenRouter requires prefixed names like ``anthropic/claude-sonnet-4-20250514``
    or ``openai/gpt-4o``.  Bare OpenAI names (``gpt-4o``) work without prefix.
    """
    if "/" in model_name:
        return model_name

    lower = model_name.lower()
    if "claude" in lower:
        return f"anthropic/{model_name}"
    if "gemini" in lower:
        return f"google/{model_name}"
    return model_name


def _openrouter_fallback(model_name: str) -> Any:
    """Route any model through OpenRouter via ``llama_index.llms.openrouter.OpenRouter``.

    Uses the dedicated LlamaIndex OpenRouter integration which handles
    all models (Claude, GPT, Gemini, etc.) natively — no validation
    patches or protocol hacks needed.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    from llama_index.llms.openrouter import OpenRouter

    return OpenRouter(
        model=_openrouter_model_name(model_name),
        api_key=api_key,
        is_function_calling_model=True,
    )


def _no_key_error(provider: str, model_name: str) -> ValueError:
    return ValueError(
        f"No API key found for '{model_name}'. "
        f"Set {provider} or OPENROUTER_API_KEY."
    )


def build_llamaindex_llm(model_name: str) -> Any:
    """Create a LlamaIndex LLM object from a model name string.

    Provider detection mirrors ``model_factory.create_model_from_string``
    but produces LlamaIndex LLM objects instead of LangChain ones.

    Fallback chain: native API key → OpenRouter.

    Supported providers:
    - ``bedrock/`` prefix → Anthropic with AWS credentials
    - ``openai/`` prefix or ``gpt-*`` / ``o1-*`` / ``o3-*`` → OpenAI
    - ``anthropic/`` prefix or ``claude`` keyword → Anthropic
    - ``google/`` prefix or ``gemini`` keyword → Gemini
    - Default → OpenAI if key available, else OpenRouter

    Model naming:
        Native provider APIs and OpenRouter use **different model name
        formats**.  Pass the name that matches the provider you intend to
        use (or that will be selected by the fallback chain):

        - **Native Anthropic**: full date-suffixed IDs, e.g.
          ``claude-sonnet-4-20250514``, ``claude-haiku-4-5-20251001``.
        - **OpenRouter**: short names, e.g. ``claude-sonnet-4``,
          ``claude-haiku-4.5``.  OpenRouter also accepts its own prefixed
          format (``anthropic/claude-sonnet-4``).
        - **Native OpenAI**: bare names work for both native and
          OpenRouter, e.g. ``gpt-4o-mini``, ``gpt-4o``.
    """

    def _log_created(llm: Any, provider: str, via: str = "native") -> Any:
        print(
            f"  [build_llamaindex_llm] {model_name} -> {type(llm).__name__} (via {via})"
        )
        return llm

    # --- AWS Bedrock (Anthropic on Bedrock) ---
    if model_name.startswith("bedrock/"):
        clean_name = model_name.removeprefix("bedrock/")
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION"))
        if aws_key and aws_secret and aws_region:
            try:
                from llama_index.llms.anthropic import Anthropic

                return _log_created(
                    Anthropic(
                        model=clean_name,
                        aws_region=aws_region,
                        aws_access_key_id=aws_key,
                        aws_secret_access_key=aws_secret,
                    ),
                    "Bedrock",
                    "aws",
                )
            except ImportError:
                pass
        fallback = _openrouter_fallback(model_name)
        if fallback:
            return _log_created(fallback, "Bedrock", "openrouter")
        raise _no_key_error(
            "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_DEFAULT_REGION",
            model_name,
        )

    # --- OpenAI ---
    if model_name.startswith("openai/") or any(
        model_name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")
    ):
        clean_name = model_name.removeprefix("openai/")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from llama_index.llms.openai import OpenAI

            return _log_created(OpenAI(model=clean_name, api_key=openai_key), "OpenAI")
        fallback = _openrouter_fallback(clean_name)
        if fallback:
            return _log_created(fallback, "OpenAI", "openrouter")
        raise _no_key_error("OPENAI_API_KEY", model_name)

    # --- Google / Gemini ---
    if model_name.startswith("google/") or "gemini" in model_name.lower():
        clean_name = model_name.removeprefix("google/")
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                from llama_index.llms.gemini import Gemini

                return _log_created(
                    Gemini(model=clean_name, api_key=google_key), "Gemini"
                )
            except ImportError:
                pass
        fallback = _openrouter_fallback(clean_name)
        if fallback:
            return _log_created(fallback, "Gemini", "openrouter")
        raise _no_key_error("GOOGLE_API_KEY", model_name)

    # --- Anthropic / Claude ---
    if model_name.startswith("anthropic/") or "claude" in model_name.lower():
        clean_name = model_name.removeprefix("anthropic/")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                from llama_index.llms.anthropic import Anthropic

                return _log_created(
                    Anthropic(model=clean_name, api_key=anthropic_key), "Anthropic"
                )
            except ImportError:
                pass
        fallback = _openrouter_fallback(clean_name)
        if fallback:
            return _log_created(fallback, "Anthropic", "openrouter")
        raise _no_key_error("ANTHROPIC_API_KEY", model_name)

    # --- Default: try native OpenAI, then OpenRouter ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from llama_index.llms.openai import OpenAI

        return _log_created(OpenAI(model=model_name, api_key=openai_key), "OpenAI")

    fallback = _openrouter_fallback(model_name)
    if fallback:
        return _log_created(fallback, "default", "openrouter")
    raise _no_key_error("OPENAI_API_KEY", model_name)


def is_llamaindex_agent(agent: Any) -> bool:
    """Check if an agent is a LlamaIndex agent (FunctionAgent, AgentWorkflow, etc.)."""
    module = getattr(type(agent), "__module__", "") or ""
    return module.startswith("llama_index") and hasattr(agent, "run")


def is_llamaindex_llm(llm: Any) -> bool:
    """Check if an LLM object is from LlamaIndex."""
    module = getattr(type(llm), "__module__", "") or ""
    return module.startswith("llama_index")


def sync_llamaindex_agents(
    agents: List[Any],
    new_llm: Any,
) -> None:
    """Push a new LLM into registered LlamaIndex agents.

    Args:
        agents: LlamaIndex agent instances (e.g. FunctionAgent).
        new_llm: The new LLM object to assign to each agent.
    """
    for agent in agents:
        agent.llm = new_llm
        logger.debug(
            "  [sync] llamaindex agent %s → %s",
            type(agent).__name__,
            getattr(new_llm, "model", "?"),
        )


def _build_llamaindex_llm(model_name: str, original_llm: Any = None) -> Any:
    """Create a fresh LlamaIndex LLM with a different model name.

    Delegates to ``build_llamaindex_llm`` which picks the correct
    provider class based on the model name and available API keys.
    """
    return build_llamaindex_llm(model_name)


def sync_llamaindex_workflow_agents(
    workflow: Any,
    proxies: list,
    combo: tuple | list,
    get_model_name: Callable[[Any], str],
) -> None:
    """Clone sub-agents with fresh LLMs and replace the workflow's agent list.

    Unlike sync_llamaindex_agents (which mutates in place), this function
    creates independent copies of each sub-agent so that parallel clones
    don't share sub-agent objects.

    Supported patterns:
    * 1 proxy → all agents (shared-LLM broadcast).
    * N proxies = N agents (positional mapping).

    Args:
        workflow: A cloned AgentWorkflow.
        proxies: The ModelProxy instances being evaluated.
        combo: The model specs corresponding to each proxy.
        get_model_name: Callable to extract a display name from a model spec.
    """
    # workflow.agents is a dict: {name: FunctionAgent, ...}
    agents_dict = workflow.agents
    agent_names = list(agents_dict.keys())
    agent_list = list(agents_dict.values())
    n_proxies = len(proxies)
    n_agents = len(agent_list)

    cloned_agents = {}

    if n_proxies == 1:
        # Shared-LLM: every sub-agent gets the same new model
        model_spec = combo[0]
        model_name = (
            model_spec if isinstance(model_spec, str) else get_model_name(model_spec)
        )
        fresh_llm = _build_llamaindex_llm(model_name)

        for name, ag in agents_dict.items():
            cloned_ag = ag.model_copy(update={"llm": fresh_llm}, deep=False)
            cloned_agents[name] = cloned_ag
            print(f"  [combo] {name} -> {model_name} ({type(fresh_llm).__name__})")
    elif n_proxies == n_agents:
        # Positional mapping: proxy i → agent i
        for (name, ag), model_spec in zip(agents_dict.items(), combo):
            model_name = (
                model_spec
                if isinstance(model_spec, str)
                else get_model_name(model_spec)
            )
            fresh_llm = _build_llamaindex_llm(model_name)
            cloned_ag = ag.model_copy(update={"llm": fresh_llm}, deep=False)
            cloned_agents[name] = cloned_ag
            print(f"  [combo] {name} -> {model_name} ({type(fresh_llm).__name__})")
    else:
        logger.warning(
            "Cannot map %d proxies to %d LlamaIndex agents. "
            "Model swap may not propagate to all agents.",
            n_proxies,
            n_agents,
        )
        return

    # Replace the workflow's agents dict with independent clones
    workflow.agents = cloned_agents


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class LlamaIndexAdapter:
    """Adapter for LlamaIndex ``FunctionAgent`` and ``AgentWorkflow`` agents."""

    invoke_method_name = "run"

    def detect(self, agent: Any) -> bool:
        return is_llamaindex_agent(agent)

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
            return future.result()

        return _invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register closures that push a new LLM into LlamaIndex agents."""
        is_workflow = hasattr(agent, "agents")  # AgentWorkflow has .agents dict

        if not is_workflow:
            # Simple FunctionAgent — direct llm assignment.
            def _sync(new_llm: Any, _agent: Any = agent) -> None:
                sync_llamaindex_agents([_agent], new_llm)

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
                sync_llamaindex_agents(_agents, new_llm)

            proxy._add_sync(_sync)

        elif n_proxies == n_agents:
            idx = all_proxies.index(proxy)
            sub_ag = agent_list[idx]

            def _sync(new_llm: Any, _ag: Any = sub_ag) -> None:
                sync_llamaindex_agents([_ag], new_llm)

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
            fresh_llm = _build_llamaindex_llm(model_name)
            for name, ag in agents_dict.items():
                cloned_agents[name] = ag.model_copy(
                    update={"llm": fresh_llm}, deep=False
                )
                print(f"  [combo] {name} -> {model_name} ({type(fresh_llm).__name__})")
        elif n_proxies == n_agents:
            for (name, ag), model_spec in zip(agents_dict.items(), combo):
                model_name = (
                    model_spec
                    if isinstance(model_spec, str)
                    else get_model_name(model_spec)
                )
                fresh_llm = _build_llamaindex_llm(model_name)
                cloned_agents[name] = ag.model_copy(
                    update={"llm": fresh_llm}, deep=False
                )
                print(f"  [combo] {name} -> {model_name} ({type(fresh_llm).__name__})")
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
