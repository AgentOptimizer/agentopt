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
from .routing import RandomRouter, Router
from .model_selection import (
    BaseModelSelector,
    DatapointResult,
    GittinsModelSelector,
    GittinsRecommendation,
    GittinsSelectionResults,
    MatrixUCBModelSelector,
    RandomSearchModelSelector,
    RandomSelectionResults,
    ModelResult,
    SelectionResults,
)

# Bayesian is optional (requires torch/botorch)
try:
    from .model_selection import BayesianOptimizationModelSelector
except ImportError:
    BayesianOptimizationModelSelector = None

_METHODS = {
    "auto": GittinsModelSelector,
    "gittins": GittinsModelSelector,
    "random": RandomSearchModelSelector,
    "ucb": MatrixUCBModelSelector,
    "matrix_ucb": MatrixUCBModelSelector,
    "bayesian": BayesianOptimizationModelSelector,
    "random_configurations": RandomSearchModelSelector,
    "random_questions": RandomSearchModelSelector,
}
_MODE_METHODS = {
    "single": ("auto", "gittins", "random", "random_configurations", "random_questions", "ucb", "matrix_ucb", "bayesian"),
    "multi": ("auto", "gittins", "random_configurations", "random_questions"),
}


def ModelSelector(
    agent=None, models=None, eval_fn=None, dataset=None, method="auto",
    *, objective_mode="single", **kwargs,
):
    """Create a selector for a single score or an accuracy/cost frontier.

    ``auto`` uses Gittins in both objective modes. Both modes support
    ``random_configurations`` (complete random configurations) and
    ``random_questions`` (shared random questions across all configurations).
    Single mode also supports ``random`` as an alias for configurations,
    ``ucb``, and ``bayesian``.

    Gittins single uses ``cost_scaling_factor=1e-4`` to price search evaluations
    inside its stopping problem. The separate ``lambda_cost`` and
    ``lambda_latency`` parameters optionally penalize deployment cost and
    latency in a scalar score; they default to zero and require single mode.
    Algorithm-specific arguments are passed through to the selected class.
    """
    if objective_mode not in _MODE_METHODS:
        raise ValueError("objective_mode must be 'single' or 'multi'")
    supported = _MODE_METHODS[objective_mode]
    if method not in supported:
        raise ValueError(
            f"Method {method!r} does not support objective_mode={objective_mode!r}. "
            f"Choose from: {', '.join(supported)}"
        )
    cls = _METHODS[method]
    if cls is None:
        raise ImportError('Bayesian optimization requires `pip install "agentopt-py[bayesian]"`')
    if cls is GittinsModelSelector:
        kwargs["objective_mode"] = objective_mode
    elif cls is RandomSearchModelSelector:
        kwargs["objective_mode"] = objective_mode
        kwargs["sampling"] = "questions" if method == "random_questions" else "configurations"
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
    "GittinsModelSelector",
    "GittinsRecommendation",
    "GittinsSelectionResults",
    "MatrixUCBModelSelector",
    "RandomSearchModelSelector",
    "RandomSelectionResults",
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
    "RandomRouter",
]
