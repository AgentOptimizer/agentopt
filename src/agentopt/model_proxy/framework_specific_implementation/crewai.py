"""CrewAI-specific LLM builder, agent sync, and FrameworkAdapter."""

import logging
from typing import Any, Callable, List, Optional

from ..adapter import FrameworkAdapter
from ..constants import MODEL_FIELDS
from ..model_copy import clone_model_spec
from ..token_tracking import TokenAccumulator

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
        # Maps id(cloned_crew) → TokenAccumulator for parallel clones.
        self._clone_trackers: dict = {}

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
            model = self._get_effective_model()
            tracker = self._get_effective_tracker()
            if tracker is not None:
                # Resolve model name for per-model token tracking.
                mname = "unknown"
                for field in MODEL_FIELDS:
                    val = getattr(model, field, None)
                    if val is not None:
                        mname = str(val)
                        break
                try:
                    before = model.get_token_usage_summary()
                    b_in = getattr(before, "prompt_tokens", 0)
                    b_out = getattr(before, "completion_tokens", 0)
                except Exception:
                    b_in = b_out = 0
                result = model.call(*args, **kwargs)
                try:
                    after = model.get_token_usage_summary()
                    tracker.add(
                        getattr(after, "prompt_tokens", 0) - b_in,
                        getattr(after, "completion_tokens", 0) - b_out,
                        model_name=mname,
                    )
                except Exception:
                    pass
                return result
            return model.call(*args, **kwargs)

        proxy_cls.call = _crewai_call

    def detect(self, agent: Any) -> bool:
        return (type(agent).__module__ or "").startswith("crewai") and hasattr(
            agent, "agents"
        )

    def get_invoke_fn(self, agent: Any) -> Callable:
        return agent.kickoff

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        pass  # Model swapping is handled transparently via BaseLLM ABC registration.

    def create_token_tracker(self, agent: Any = None) -> TokenAccumulator:
        """Return a TokenAccumulator for this agent.

        Token tracking is handled by ``_crewai_call`` on each proxy, which
        reads the LLM's ``_token_usage`` delta per call. No kickoff patching
        needed — just return the right accumulator.
        """
        # Parallel clone path: retrieve pre-created tracker.
        if agent is not None:
            tracker = self._clone_trackers.pop(id(agent), None)
            if tracker is not None:
                return tracker

        # Sequential path: bare accumulator (proxies get it via _set_token_tracker).
        return TokenAccumulator()

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
        from ..proxy import ModelProxy

        cloned = agent.model_copy(deep=False)

        assert self.detect(
            cloned
        ), f"clone_for_parallel called on non-Crew: {type(cloned).__name__}"

        crew_agents = cloned.agents
        n_proxies = len(proxies)
        n_agents = len(crew_agents)
        cloned_agents = []

        # Shared tracker for all fresh proxies in this clone.
        tracker = TokenAccumulator()

        if n_proxies == 1:
            # Shared-LLM: every sub-agent gets the same new model
            model_spec = combo[0]
            model_name = get_model_name(model_spec)
            if isinstance(model_spec, str):
                fresh_llm = build_crewai_llm(model_name)
            else:
                fresh_llm = clone_model_spec(model_spec)
            fresh_proxy = ModelProxy(fresh_llm)
            fresh_proxy._set_token_tracker(tracker)
            for ag in crew_agents:
                cloned_ag = ag.model_copy(update={"llm": fresh_proxy}, deep=False)
                cloned_agents.append(cloned_ag)
                logger.debug(
                    "  [clone] %s → %s",
                    ag.role if hasattr(ag, "role") else "agent",
                    model_name,
                )
        elif n_proxies == n_agents:
            # Positional mapping: proxy i → agent i
            for ag, model_spec in zip(crew_agents, combo):
                model_name = get_model_name(model_spec)
                if isinstance(model_spec, str):
                    fresh_llm = build_crewai_llm(model_name)
                else:
                    fresh_llm = clone_model_spec(model_spec)
                fresh_proxy = ModelProxy(fresh_llm)
                fresh_proxy._set_token_tracker(tracker)
                cloned_ag = ag.model_copy(update={"llm": fresh_proxy}, deep=False)
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

        # Store tracker for create_token_tracker to retrieve.
        self._clone_trackers[id(cloned)] = tracker

        return cloned


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

register_adapter(CrewAIAdapter())
