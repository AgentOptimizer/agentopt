"""CrewAI-specific LLM builder, agent sync, and FrameworkAdapter."""

import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


def _model_name_from_llm(llm: Any) -> Optional[str]:
    """Extract model name string from any LLM object."""
    from .constants import MODEL_FIELDS

    return next(
        (str(getattr(llm, f)) for f in MODEL_FIELDS if hasattr(llm, f)), None
    )


def build_crewai_llm(model_name: str) -> Optional[Any]:
    """Create a CrewAI LLM from a model name string.

    Returns a new ``crewai.LLM`` instance, or ``None`` on failure.
    """
    from crewai import LLM  # type: ignore[import-untyped]

    return LLM(model=model_name)


def sync_crew_agents(
    agent: Any,
    proxies: list,
    combo: tuple | list,
    get_model_name: Callable[[Any], str],
) -> None:
    """Push model changes into CrewAI Crew's sub-agents.

    Frameworks like CrewAI copy the LLM at construction time (via
    ``create_llm``), so ``proxy.set_model()`` alone has no effect on
    the agent.  This function directly updates the agent's internal LLM
    objects after a proxy swap.

    Supported patterns:
    * 1 proxy → all agents (shared-LLM broadcast).
    * N proxies = N agents (positional mapping).

    Args:
        agent: The top-level agent (e.g. a CrewAI ``Crew``).
        proxies: The ``ModelProxy`` instances being swapped.
        combo: The model specs corresponding to each proxy.
        get_model_name: Callable to extract a display name from a model spec.
    """
    from .constants import is_crewai_crew

    assert is_crewai_crew(
        agent
    ), f"sync_crew_agents called on non-Crew agent: {type(agent).__name__}"

    crew_agents = agent.agents
    n_proxies = len(proxies)
    n_agents = len(crew_agents)

    if n_proxies == 1:
        # Shared-LLM: every crew agent gets the same new model
        model_spec = combo[0]
        model_name = (
            model_spec if isinstance(model_spec, str) else get_model_name(model_spec)
        )
        for ag in crew_agents:
            ag.llm = build_crewai_llm(model_name)
            logger.debug(
                "  [sync] %s → %s",
                ag.role if hasattr(ag, "role") else "agent",
                model_name,
            )
    elif n_proxies == n_agents:
        # Positional mapping: proxy i → agent i
        for ag, model_spec in zip(crew_agents, combo):
            model_name = (
                model_spec
                if isinstance(model_spec, str)
                else get_model_name(model_spec)
            )
            ag.llm = build_crewai_llm(model_name)
            logger.debug(
                "  [sync] %s → %s",
                ag.role if hasattr(ag, "role") else "agent",
                model_name,
            )
    else:
        logger.warning(
            "Cannot map %d proxies to %d crew agents. "
            "Model swap may not propagate to all agents.",
            n_proxies,
            n_agents,
        )


def clone_crew_agents(
    crew: Any,
    proxies: list,
    combo: tuple | list,
    get_model_name: Callable[[Any], str],
) -> None:
    """Clone sub-agents with fresh LLMs and replace the crew's agents list.

    Unlike sync_crew_agents (which mutates in place), this function creates
    independent copies of each sub-agent so that parallel clones don't share
    sub-agent objects.

    Supported patterns:
    * 1 proxy → all agents (shared-LLM broadcast).
    * N proxies = N agents (positional mapping).

    Args:
        crew: A cloned CrewAI Crew.
        proxies: The ModelProxy instances being evaluated.
        combo: The model specs corresponding to each proxy.
        get_model_name: Callable to extract a display name from a model spec.
    """
    from .constants import is_crewai_crew

    assert is_crewai_crew(
        crew
    ), f"clone_crew_agents called on non-Crew: {type(crew).__name__}"

    crew_agents = crew.agents
    n_proxies = len(proxies)
    n_agents = len(crew_agents)
    cloned_agents = []

    if n_proxies == 1:
        # Shared-LLM: every sub-agent gets the same new model
        model_name = combo[0] if isinstance(combo[0], str) else get_model_name(combo[0])
        fresh_llm = build_crewai_llm(model_name)
        for ag in crew_agents:
            cloned_ag = ag.model_copy(update={"llm": fresh_llm}, deep=False)
            cloned_agents.append(cloned_ag)
            logger.debug(
                "  [clone] %s → %s",
                ag.role if hasattr(ag, "role") else "agent",
                model_name,
            )
    elif n_proxies == n_agents:
        # Positional mapping: proxy i → agent i
        for ag, model_spec in zip(crew_agents, combo):
            model_name = (
                model_spec
                if isinstance(model_spec, str)
                else get_model_name(model_spec)
            )
            fresh_llm = build_crewai_llm(model_name)
            cloned_ag = ag.model_copy(update={"llm": fresh_llm}, deep=False)
            cloned_agents.append(cloned_ag)
            logger.debug(
                "  [clone] %s → %s",
                ag.role if hasattr(ag, "role") else "agent",
                model_name,
            )
    else:
        logger.warning(
            "Cannot map %d proxies to %d crew agents. "
            "Model swap may not propagate to all agents.",
            n_proxies,
            n_agents,
        )
        return

    crew.agents = cloned_agents


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class CrewAIAdapter:
    """Adapter for CrewAI ``Crew`` agents."""

    invoke_method_name = "kickoff"

    def detect(self, agent: Any) -> bool:
        from .constants import is_crewai_crew

        return is_crewai_crew(agent)

    def get_invoke_fn(self, agent: Any) -> Callable:
        return agent.kickoff

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register closures that push model changes into the crew's sub-agents."""
        crew_agents = agent.agents
        n_proxies = len(all_proxies)
        n_agents = len(crew_agents)

        if n_proxies == 1:
            # Shared-LLM: every sub-agent gets the same model.
            def _sync(new_llm: Any, _agents: list = list(crew_agents)) -> None:
                name = _model_name_from_llm(new_llm)
                if name:
                    fresh = build_crewai_llm(name)
                    for ag in _agents:
                        ag.llm = fresh

            proxy._add_sync(_sync)

        elif n_proxies == n_agents:
            # Positional mapping: proxy i → sub-agent i.
            idx = all_proxies.index(proxy)
            sub_ag = crew_agents[idx]

            def _sync(new_llm: Any, _ag: Any = sub_ag) -> None:
                name = _model_name_from_llm(new_llm)
                if name:
                    _ag.llm = build_crewai_llm(name)

            proxy._add_sync(_sync)

        else:
            logger.warning(
                "CrewAIAdapter: cannot map %d proxies to %d crew agents — "
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
        """Shallow-copy the Crew, then independently clone each sub-agent."""
        cloned = agent.model_copy(deep=False)
        clone_crew_agents(cloned, proxies, combo, get_model_name)
        return cloned


# Self-register — runs when this module is first imported.
from .adapter import register_adapter  # noqa: E402

register_adapter(CrewAIAdapter())
