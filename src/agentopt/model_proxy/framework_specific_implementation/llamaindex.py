"""LlamaIndex-specific agent sync.

LlamaIndex FunctionAgents use strict Pydantic validation, so ModelProxy
cannot be passed as ``llm=`` directly. Instead, the agent is created with
a real LLM and ``register_llamaindex_agent()`` ensures that
``agent.llm`` is updated whenever ``proxy.set_model()`` is called.
"""

import logging
from typing import Any, Callable, List

from pydantic import BaseModel

from ..constants import MODEL_FIELDS

logger = logging.getLogger(__name__)


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


def _build_llamaindex_llm(model_name: str, original_llm: Any) -> Any:
    """Create a fresh LlamaIndex LLM with a different model name.

    Uses model_copy on the original LLM to preserve API keys and settings.
    """
    if isinstance(original_llm, BaseModel):
        target_field = next(
            (f for f in MODEL_FIELDS if f in type(original_llm).model_fields),
            None,
        )
        if target_field:
            return original_llm.model_copy(update={target_field: model_name})

    raise TypeError(
        f"Cannot create LlamaIndex LLM variant of {type(original_llm).__name__}"
    )


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
        original_llm = agent_list[0].llm
        fresh_llm = _build_llamaindex_llm(model_name, original_llm)

        for name, ag in agents_dict.items():
            cloned_ag = ag.model_copy(update={"llm": fresh_llm}, deep=False)
            cloned_agents[name] = cloned_ag
            logger.debug(
                "  [sync] %s → %s",
                name,
                model_name,
            )
    elif n_proxies == n_agents:
        # Positional mapping: proxy i → agent i
        for (name, ag), model_spec in zip(agents_dict.items(), combo):
            model_name = (
                model_spec
                if isinstance(model_spec, str)
                else get_model_name(model_spec)
            )
            fresh_llm = _build_llamaindex_llm(model_name, ag.llm)
            cloned_ag = ag.model_copy(update={"llm": fresh_llm}, deep=False)
            cloned_agents[name] = cloned_ag
            logger.debug(
                "  [sync] %s → %s",
                name,
                model_name,
            )
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
        """
        import asyncio
        import inspect

        method = agent.run

        def _invoke(input_data: Any) -> Any:
            async def _async_run() -> Any:
                if isinstance(input_data, dict):
                    result = method(**input_data)
                else:
                    result = method(input_data)
                if inspect.isawaitable(result):
                    return await result
                return result

            return asyncio.run(_async_run())

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
        """Shallow-copy the workflow, then independently clone each sub-agent."""
        cloned = agent.model_copy(deep=False) if isinstance(agent, BaseModel) else agent
        if hasattr(agent, "agents"):
            # AgentWorkflow — replace the agents dict with fresh clones.
            sync_llamaindex_workflow_agents(cloned, proxies, combo, get_model_name)
        return cloned


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

register_adapter(LlamaIndexAdapter())
