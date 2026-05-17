"""Policy registry + resolver for daemon-side routing.

The daemon reconstructs a :class:`Router` instance from a
``{"policy": <name>, "kwargs": {...}}`` dict that arrives via either
the ``agentopt serve`` CLI flags or the ``router`` field of
``POST /sessions``.

Two policy-name conventions resolve here:

* **Short aliases** for built-in policies, listed in
  :data:`BUILTIN_POLICIES`.  E.g. ``"random"`` → :class:`RandomRouter`.
* **``module:Class``** dotted paths for everything else.  The module
  must already be importable (a future ``--policy-module`` flag will
  pre-import user modules so custom routers resolve too).

Built-in policies are the only ones supported in v1's daemon mode.
Custom routers raise on the wire today; the dispatch path is in place
so adding the plugin loader later is a small change.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Type

from .base import Router


# ---------------------------------------------------------------------------
# Built-in registry — short aliases the daemon recognises
# ---------------------------------------------------------------------------


BUILTIN_POLICIES: Dict[str, str] = {
    "random": "agentopt.routing.random_policy:RandomRouter",
    # Future policies (length-based, classifier, bandit) register here so
    # they get short CLI aliases and YAML/JSON config names for free.
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_class(policy: str) -> Type[Router]:
    """Look up the Router subclass referenced by *policy*.

    Accepts short aliases (``"random"``) or ``"module:Class"`` paths.
    Raises :class:`ValueError` with a clear message if the policy is
    unknown or the import fails.
    """
    target = BUILTIN_POLICIES.get(policy, policy)
    if ":" not in target:
        raise ValueError(
            f"Unknown routing policy {policy!r}.  Known built-ins: "
            f"{sorted(BUILTIN_POLICIES)}.  For a custom policy, use "
            f'"module:Class" syntax with the module on the daemon\'s '
            "PYTHONPATH."
        )
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Could not import routing policy module {module_name!r}: {exc}. "
            "Custom routers must be importable on the daemon's PYTHONPATH."
        ) from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module {module_name!r} has no attribute {class_name!r}."
        ) from exc
    if not isinstance(cls, type) or not issubclass(cls, Router):
        raise ValueError(
            f"{module_name}:{class_name} is not a subclass of "
            "agentopt.routing.Router."
        )
    return cls


def resolve_policy(policy: str, kwargs: Dict[str, Any]) -> Router:
    """Instantiate the policy named *policy* with *kwargs*.

    Used by the daemon (both at startup for ``--routing-policy`` and
    per-request when ``POST /sessions`` carries a ``router`` field).
    """
    cls = _resolve_class(policy)
    return cls.from_config(**kwargs)
