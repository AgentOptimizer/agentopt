"""
agentopt — Framework-agnostic LLM model selection optimizer for agents.

Define an agent class with ``__init__(self, models)`` and
``run(self, input_data)``, then let a ModelSelector find the best
model combination.
"""

__version__ = "0.1.0"

from dataclasses import dataclass
from typing import Optional

from agentopt.proxy import CallRecord, LLMTracker, SessionInfo


@dataclass(frozen=True)
class SessionProxy:
    """Proxy connection details for the current tracking session.

    Returned by :func:`get_current_session_proxy`. Subprocess agents use
    these primitives to route LLM traffic through agentopt's tracking proxy.

    For tools that respect ``HTTPS_PROXY`` env vars, call :meth:`env_dict`
    for a ready-to-spread mapping. For tools like OpenClaw that ignore
    env vars and need config-file injection, read ``url`` and ``ca_pem``
    directly.
    """

    url: str
    port: int
    ca_pem: str
    ca_bundle_path: str

    def env_dict(self) -> dict:
        """Env-var-shaped form for ``HTTPS_PROXY``-respecting subprocesses.

        Spread into ``subprocess.run`` env::

            proxy = agentopt.get_current_session_proxy()
            env = {**os.environ, **(proxy.env_dict() if proxy else {})}
            subprocess.run(["gemini", "cli", ...], env=env)
        """
        return {
            "HTTPS_PROXY": self.url,
            "SSL_CERT_FILE": self.ca_bundle_path,
            "REQUESTS_CA_BUNDLE": self.ca_bundle_path,
            "NODE_EXTRA_CA_CERTS": self.ca_bundle_path,
        }


def get_current_session_proxy() -> Optional[SessionProxy]:
    """Return proxy details for the current tracking session, or ``None``.

    Each :meth:`LLMTracker.track` scope has its own mitmproxy port
    (via ContextVar), so this is safe for parallel evaluation. Returns
    ``None`` when no tracking session is active.
    """
    from agentopt.proxy.interceptor import _active_session_var
    from agentopt.proxy.tracker import _MITMPROXY_CA_CERT, _ensure_ca_bundle

    active = _active_session_var.get()
    if active is None:
        return None

    with open(_MITMPROXY_CA_CERT) as f:
        ca_pem = f.read().strip()

    return SessionProxy(
        url=f"http://127.0.0.1:{active.port}",
        port=active.port,
        ca_pem=ca_pem,
        ca_bundle_path=_ensure_ca_bundle(),
    )


from .base_models import AgentFn, Dataset, EvalFn, ModelsConfig
from .routing import RandomRouter, RouteContext, RouteDecision, Router
from .model_selection import (
    ArmEliminationModelSelector,
    BaseModelSelector,
    BruteForceModelSelector,
    EpsilonLUCBModelSelector,
    HillClimbingModelSelector,
    LMProposalModelSelector,
    DatapointResult,
    MatrixUCBLRFModelSelector,
    MatrixUCBModelSelector,
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
    "matrix_ucb": MatrixUCBModelSelector,
    "matrix_ucb_lrf": MatrixUCBLRFModelSelector,
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
            ``"epsilon_lucb"``, ``"matrix_ucb"``, ``"matrix_ucb_lrf"``,
            ``"threshold"``,
            ``"lm_proposal"``, ``"bayesian"``.
        **kwargs: Additional arguments passed to the selector
            (e.g. ``epsilon``, ``threshold``, ``sample_fraction``, ``warmup_fraction``
            for matrix UCB-LRF).

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
    "MatrixUCBModelSelector",
    "MatrixUCBLRFModelSelector",
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
    "SessionProxy",
    "get_current_session_proxy",
    # Routing
    "Router",
    "RouteContext",
    "RouteDecision",
    "RandomRouter",
]
