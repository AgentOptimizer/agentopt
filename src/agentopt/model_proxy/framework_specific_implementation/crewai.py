"""CrewAI-specific LLM builder, agent sync, and FrameworkAdapter."""

import logging
from typing import Any, Callable, Dict, List, Tuple

from ..adapter import FrameworkAdapter

logger = logging.getLogger(__name__)


def is_crewai_llm(llm: Any) -> bool:
    """Check if an LLM object is from CrewAI."""
    return (type(llm).__module__ or "").startswith("crewai")


def build_crewai_llm(model_name: str) -> Optional[Any]:
    """Create a CrewAI LLM from a model name string.

    Returns a new ``crewai.LLM`` instance, or ``None`` on failure.
    """
    from crewai import LLM  # type: ignore[import-untyped]

    return LLM(model=model_name)


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class CrewAIAdapter(FrameworkAdapter):
    """Adapter for CrewAI ``Crew`` agents."""

    invoke_method_name = "kickoff"

    def __init__(self) -> None:
        # Maps agent id → (input_tokens_seen, output_tokens_seen) at last get_token_usage call.
        self._token_baseline: Dict[int, Tuple[int, int]] = {}

    @classmethod
    def patch_proxy_class(cls, proxy_cls: type) -> None:
        """Register *proxy_cls* as a virtual subclass of CrewAI's ``BaseLLM``
        and patch ``call()`` to delegate to the wrapped model.

        1. ``BaseLLM.register(proxy_cls)`` — makes ``isinstance(proxy, BaseLLM)``
           return ``True``, so CrewAI's ``create_llm()`` returns the proxy as-is.
        2. Attaches ``call`` so CrewAI's agent executor routes through the proxy.
        """
        from crewai.llms.base_llm import BaseLLM  # type: ignore[import-untyped]

        BaseLLM.register(proxy_cls)

        def _crewai_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = object.__getattribute__(self, "_optmodel")
            return model.call(*args, **kwargs)

        proxy_cls.call = _crewai_call

    def detect(self, agent: Any) -> bool:
        return (type(agent).__module__ or "").startswith("crewai") and hasattr(
            agent, "agents"
        )

    def get_invoke_fn(self, agent: Any) -> Callable:
        return agent.kickoff

    def get_token_usage(self, agent: Any) -> Tuple[int, int]:
        """Return tokens consumed since the last call, via crew.usage_metrics."""
        metrics = agent.usage_metrics
        if metrics is None:
            return (0, 0)
        total_in = metrics.prompt_tokens
        total_out = metrics.completion_tokens
        key = id(agent)
        prev_in, prev_out = self._token_baseline.get(key, (0, 0))
        self._token_baseline[key] = (total_in, total_out)
        return (total_in - prev_in, total_out - prev_out)

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Clone sub-agents with fresh LLMs and replace the crew's agents list.

        Unlike the old sync_crew_agents (which mutated in place), this function
        creates independent copies of each sub-agent so that parallel clones
        don't share sub-agent objects.

        Supported patterns:
        * 1 proxy → all agents (shared-LLM broadcast).
        * N proxies = N agents (positional mapping).

        Args:
            agent: A CrewAI Crew to clone.
            proxies: The ModelProxy instances being evaluated.
            combo: The model specs corresponding to each proxy.
            get_model_name: Callable to extract a display name from a model spec.
        """
        cloned = agent.model_copy(deep=False)

        assert self.detect(
            cloned
        ), f"clone_for_parallel called on non-Crew: {type(cloned).__name__}"

        crew_agents = cloned.agents
        n_proxies = len(proxies)
        n_agents = len(crew_agents)
        cloned_agents = []

        if n_proxies == 1:
            # Shared-LLM: every sub-agent gets the same new model
            model_name = (
                combo[0] if isinstance(combo[0], str) else get_model_name(combo[0])
            )
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

        cloned.agents = cloned_agents

        # Clone tasks and remap task.agent to the cloned agents.
        # Must use `is` (identity) instead of dict lookup — Pydantic models
        # aren't hashable.  Must clone tasks — model_copy(deep=False) shares
        # Task objects with the original crew, so direct mutation would
        # corrupt subsequent clones.
        cloned_tasks = []
        for task in cloned.tasks:
            new_agent = task.agent
            for old_ag, new_ag in zip(crew_agents, cloned_agents):
                if task.agent is old_ag:
                    new_agent = new_ag
                    break
            cloned_tasks.append(
                task.model_copy(update={"agent": new_agent}, deep=False)
            )
        cloned.tasks = cloned_tasks

        return cloned


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

register_adapter(CrewAIAdapter())
