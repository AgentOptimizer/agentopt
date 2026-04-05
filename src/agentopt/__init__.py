"""
agentopt — Framework-agnostic LLM model selection optimizer for agents.

Define an agent class with ``__init__(self, models)`` and
``run(self, input_data)``, then let a ModelSelector find the best
model combination.
"""

__version__ = "0.1.0"

from agentopt.proxy import CallRecord, LLMTracker, SessionInfo


def get_current_session_env() -> dict:
    """Return proxy env vars for the current tracking session.

    Use inside ``agent.run()`` when spawning subprocess-based agents::

        import agentopt, os, subprocess
        env = {**os.environ, **agentopt.get_current_session_env()}
        subprocess.run(["gemini", "cli", ...], env=env)

    Returns an empty dict if no tracking session is active.
    """
    # Lazy import to avoid circular dependency at module load time.
    from agentopt.proxy.interceptor import _proxy_base_url, _session_id_var

    session_id = _session_id_var.get()
    if session_id is None or _proxy_base_url is None:
        return {}
    url = f"{_proxy_base_url}/{session_id}"
    return {
        "OPENAI_BASE_URL": url,
        "ANTHROPIC_BASE_URL": url,
        "GOOGLE_API_BASE": url,
        "AGENTOPT_SESSION_ID": session_id,
        "AGENTOPT_PROXY_URL": _proxy_base_url,
    }


from .base_models import AgentFn, Dataset, EvalFn, ModelsConfig
from .model_selection import (
    ArmEliminationModelSelector,
    BaseModelSelector,
    BruteForceModelSelector,
    EpsilonLUCBModelSelector,
    HillClimbingModelSelector,
    LMProposalModelSelector,
    DatapointResult,
    ModelResult,
    RandomSearchModelSelector,
    SelectionResults,
    ThresholdBanditSEModelSelector,
)

# Bayesian is optional (requires torch/botorch)
try:
    from .model_selection import BayesianOptimizationModelSelector
except ImportError:
    BayesianOptimizationModelSelector = None

_METHODS = {
    "auto": ArmEliminationModelSelector,
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "epsilon_lucb": EpsilonLUCBModelSelector,
    "threshold": ThresholdBanditSEModelSelector,
    "lm_proposal": LMProposalModelSelector,
    "bayesian": BayesianOptimizationModelSelector,
}


def ModelSelector(
    agent=None, models=None, eval_fn=None, dataset=None, method="auto", **kwargs,
):
    """Create a model selector.

    Convenience wrapper that dispatches to the right selector class
    based on ``method``.

    Args:
        agent: Agent class with ``__init__(self, models)`` and
            ``run(self, input_data)``.
        models: Dict mapping step names to lists of candidate models.
        eval_fn: Scoring function ``(expected, actual) -> float``.
        dataset: List of ``(input_data, expected_output)`` pairs.
        method: Selection algorithm. ``"auto"`` (default) automatically finds
            the best combination (same implementation as ``"arm_elimination"`` —
            strong best-arm identification with lower search cost than
            ``"brute_force"``). Other options: ``"brute_force"``,
            ``"random"``, ``"hill_climbing"``, ``"arm_elimination"``,
            ``"epsilon_lucb"``, ``"threshold"``, ``"lm_proposal"``,
            ``"bayesian"``.
        **kwargs: Additional arguments passed to the selector
            (e.g. ``epsilon``, ``threshold``, ``sample_fraction``).

    Returns:
        A selector instance. Call ``.select_best()`` to run.
    """
    cls = _METHODS.get(method)
    if cls is None:
        if method == "bayesian":
            raise ImportError(
                "Bayesian optimization requires optional dependencies: "
                'install with `pip install "agentopt-py[bayesian]"`'
            )
        raise ValueError(
            f"Unknown method {method!r}. " f"Choose from: {', '.join(_METHODS)}"
        )
    return cls(agent=agent, models=models, eval_fn=eval_fn, dataset=dataset, **kwargs)


__all__ = [
    # Metadata
    "__version__",
    # Core API
    "ModelSelector",
    "BaseModelSelector",
    "LLMTracker",
    "CallRecord",
    # Selectors
    "BruteForceModelSelector",
    "RandomSearchModelSelector",
    "HillClimbingModelSelector",
    "ArmEliminationModelSelector",
    "EpsilonLUCBModelSelector",
    "ThresholdBanditSEModelSelector",
    "LMProposalModelSelector",
    "BayesianOptimizationModelSelector",
    # Result types
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
    # Type aliases
    "AgentFn",
    "Dataset",
    "EvalFn",
    "ModelsConfig",
    # Proxy / session helpers
    "SessionInfo",
    "get_current_session_env",
]
