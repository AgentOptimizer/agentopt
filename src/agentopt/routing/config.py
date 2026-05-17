"""Policy registry + resolver for daemon-side routing.

The daemon reconstructs a :class:`Router` instance from a
``{"policy": <name>, "kwargs": {...}}`` dict that arrives via either
the ``agentopt serve`` CLI flags or the ``router`` field of
``POST /sessions``.

Two policy-name conventions resolve here:

* **Short aliases** for built-in policies, listed in
  :data:`BUILTIN_POLICIES`.  E.g. ``"random"`` → :class:`RandomRouter`.
* **``module:Class``** dotted paths for everything else.  The named
  module must be importable on the daemon's ``sys.path``.  Built-ins
  always are; user modules need either ``PYTHONPATH`` setup or
  :func:`load_policy_module` (wired to the ``--policy-module`` CLI flag).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Type

from .base import Router

logger = logging.getLogger(__name__)


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


def load_policy_module(path: str) -> str:
    """Import a Python file by path so its ``Router`` subclasses resolve.

    Used by ``agentopt serve --policy-module path/to/my_policies.py``.
    Adds the module to ``sys.modules`` under its file stem (e.g.
    ``my_policies.py`` → ``my_policies``); ``resolve_policy`` can then
    reach the classes inside via ``"my_policies:MyRouter"``.

    Returns the module name it registered.  Raises :class:`ValueError`
    with a clear message on missing files / import errors.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"policy module {path!r} does not exist or is not a file")
    module_name = p.stem
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not build import spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        if isinstance(exc, ValueError):
            raise
        raise ValueError(
            f"Could not import policy module {path!r} as {module_name!r}: {exc}"
        ) from exc
    logger.info("Loaded policy module %s from %s", module_name, p)
    return module_name
