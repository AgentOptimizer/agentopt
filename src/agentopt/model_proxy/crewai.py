"""CrewAI-specific LLM builder and agent sync."""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


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
