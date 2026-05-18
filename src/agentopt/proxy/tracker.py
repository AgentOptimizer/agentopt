"""LLMTracker — public proxy-based LLM call tracker.

Thin delegator to a :class:`_Backend`.  Two backends ship with the
package:

* :class:`LocalBackend` — runs the mitmproxy ``SessionMaster`` in the
  current Python process (default).
* :class:`RemoteBackend` — talks to a long-lived ``agentopt serve``
  daemon over HTTP.  Selected when the ``AGENTOPT_GATEWAY_URL`` env
  var is set.

The tracker's public surface is unchanged; switching modes is a
deployment concern, not an API choice.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

from ._backend import (
    _MITMPROXY_CA_CERT,
    _MITMPROXY_CONFDIR,
    _Backend,
    _ensure_ca_bundle,
)
from ._local_backend import LocalBackend
from .models import CallRecord
from .session import SessionInfo

if TYPE_CHECKING:
    from agentopt.routing.base import Router

__all__ = [
    "LLMTracker",
    # Re-exports so ``agentopt/__init__.py`` and any external callers can
    # continue to import these helpers from ``agentopt.proxy.tracker``.
    "_MITMPROXY_CA_CERT",
    "_MITMPROXY_CONFDIR",
    "_ensure_ca_bundle",
]


class LLMTracker:
    """Tracks LLM API calls via a pluggable backend.

    Selection rule (in ``__init__``):

    * ``AGENTOPT_GATEWAY_URL`` set → :class:`RemoteBackend` (talks to a
      running ``agentopt serve`` daemon).
    * Otherwise → :class:`LocalBackend` (today's in-process proxy).

    The public surface is identical in both modes — the env var is the
    entire deployment switch::

        # Dev (default)
        python my_script.py

        # Production
        AGENTOPT_GATEWAY_URL=http://gw.team.local:9000 python my_script.py

    Usage — single session::

        with LLMTracker(combo_id="gpt4o+haiku") as tracker:
            result = agent.run(input_data)
            env = {**os.environ, **tracker.get_session_env(session)}
            subprocess.run(["gemini", "cli", ...], env=env)
        usage = tracker.get_usage(combo_id="gpt4o+haiku")

    Usage — multi-session host (``ModelSelector`` pattern)::

        with LLMTracker() as tracker:
            for combo, dp in ...:
                with tracker.track(combo_id=combo, data_id=dp.id):
                    ...
    """

    _DEFAULT_CACHE_DIR = ".agentopt_cache"
    _GATEWAY_URL_ENV = "AGENTOPT_GATEWAY_URL"

    def __init__(
        self,
        *,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        router: Optional["Router"] = None,
        cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = _DEFAULT_CACHE_DIR,
    ) -> None:
        """Create a tracker.

        Two usage shapes:

        * **Single-session sugar** — pass ``data_id`` / ``combo_id`` /
          ``agent_id`` / ``router`` and use it as a context manager::

              with LLMTracker(combo_id="X") as tracker:
                  agent.run(...)

          ``__enter__`` will ``start()`` *and* open a tracking session
          (and apply *router* if given).  ``__exit__`` closes both.

        * **Multi-session host** — pass nothing, then call
          :meth:`track` inside.  Used by ``ModelSelector`` to host
          one tracker over many ``(combo, datapoint)`` evaluations::

              with LLMTracker() as tracker:
                  for combo, dp in ...:
                      with tracker.track(combo_id=combo, data_id=dp.id):
                          ...
        """
        url = os.environ.get(self._GATEWAY_URL_ENV)
        if url:
            # Imported lazily so the local-mode-only install path doesn't
            # have to evaluate the remote backend module.
            from ._remote_backend import RemoteBackend

            self._backend: _Backend = RemoteBackend(gateway_url=url)
        else:
            self._backend = LocalBackend(cache=cache, cache_dir=cache_dir)

        # Session args stashed for the context-manager sugar.  None of
        # them set → ``__enter__`` is start-only and the user opens
        # sessions explicitly via ``self.track(...)``.
        self._auto_data_id = data_id
        self._auto_combo_id = combo_id
        self._auto_agent_id = agent_id
        self._auto_router = router

    # ------------------------------------------------------------------
    # Backend access (tests / advanced callers)
    # ------------------------------------------------------------------

    @property
    def _registry(self):
        """Local-mode provider registry.

        Exposed for tests that introspect the registry directly.  Raises
        ``AttributeError`` in remote mode (registry lives on the daemon).
        """
        return self._backend._registry  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Install the httpx redirect and prepare the backend for ``track()``."""
        self._backend.start()

    def stop(self) -> None:
        """Tear down all live sessions, restore httpx, flush cache.

        Record queries remain valid after ``stop()``; call :meth:`close`
        for full teardown (releases the remote backend's HTTP client).
        """
        self._backend.stop()

    def close(self) -> None:
        """Final teardown.  Idempotent; safe to call after ``stop``.

        Releases every resource the backend holds.  For ``RemoteBackend``
        this drops the long-lived control-plane HTTP client; for
        ``LocalBackend`` this is equivalent to ``stop`` when the tracker
        is still active and a no-op otherwise.
        """
        self._backend.close()

    # Context-manager sugar.  ``__enter__`` always ``start()``s.  If a
    # session-identifying arg (``data_id`` / ``combo_id`` / ``agent_id``)
    # was supplied to ``__init__``, it auto-opens a single tracking
    # session that ``__exit__`` will close.  A bare ``router=`` alone is
    # *not* a single-session signal — it sets a default router that
    # nested ``tracker.track()`` calls inherit, supporting the
    # multi-session host pattern with per-datapoint attribution.
    def __enter__(self) -> "LLMTracker":
        self.start()
        if any(
            arg is not None
            for arg in (self._auto_data_id, self._auto_combo_id, self._auto_agent_id,)
        ):
            self._auto_track_ctx = self.track(
                data_id=self._auto_data_id or "default",
                combo_id=self._auto_combo_id or "default",
                agent_id=self._auto_agent_id,
                router=self._auto_router,
            )
            self._auto_track_ctx.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        ctx = getattr(self, "_auto_track_ctx", None)
        if ctx is not None:
            try:
                ctx.__exit__(exc_type, exc, tb)
            finally:
                self._auto_track_ctx = None
        # Use stop() (not close()) so post-exit calls like
        # ``tracker.print_summary()`` / ``tracker.get_records()`` still
        # work — they're the natural follow-up to the ``with`` block.
        # RemoteBackend's ``__del__`` is the safety net for the http
        # client; users who want explicit release call ``tracker.close()``.
        self.stop()

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    @contextmanager
    def track(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        router: Optional["Router"] = None,
    ) -> Iterator[SessionInfo]:
        """Open a tracking session.  See :meth:`_Backend.track`.

        All attribution fields are optional — set the ones you care
        about and leave the rest.  A typical per-datapoint loop only
        sets ``data_id``::

            with LLMTracker(router=router) as tracker:
                for q in questions:
                    with tracker.track(data_id=q):   # router inherited
                        agent.run(q)

        If ``router`` is omitted, the router passed to
        :class:`LLMTracker`'s constructor (if any) is inherited.
        """
        if router is None:
            router = self._auto_router
        with self._backend.track(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id, router=router,
        ) as session:
            yield session

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def get_session_env(self, session: SessionInfo) -> Dict[str, str]:
        """Return env vars that route a subprocess through this session's port.

        Usage::

            with tracker.track(data_id="dp_1", combo_id="c") as session:
                env = {**os.environ, **tracker.get_session_env(session)}
                subprocess.run(["tb", "run", ...], env=env)
        """
        return self._backend.get_session_env(session)

    # ------------------------------------------------------------------
    # Provider registry
    # ------------------------------------------------------------------

    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        """Add or replace an LLM provider."""
        self._backend.register_provider(name, base_url, path_patterns)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def flush_cache(self) -> None:
        """Flush dirty cache entries to disk immediately."""
        self._backend.flush_cache()

    def clear_cache(self) -> None:
        """Clear all cached responses and database rows."""
        self._backend.clear_cache()

    # ------------------------------------------------------------------
    # Record queries
    # ------------------------------------------------------------------

    def get_records(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[CallRecord]:
        """Return ``CallRecord`` list, filtered by any combination of IDs."""
        return self._backend.get_records(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
        )

    @property
    def records(self) -> List[CallRecord]:
        """All ``CallRecord``s captured by this tracker.

        Equivalent to :meth:`get_records()` with no filters.  For
        single-session sugar (``with LLMTracker(combo_id="X") as t``)
        these are the records of the auto-opened session.
        """
        return self._backend.get_records()

    def print_summary(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Print model sequence, per-model tokens, and total latency.

        Filters records by the given IDs (defaults to all).  Thin wrapper
        over :func:`agentopt.routing.print_routing_summary`.
        """
        from agentopt.routing.summary import print_routing_summary

        print_routing_summary(
            self.get_records(data_id=data_id, combo_id=combo_id, agent_id=agent_id,)
        )

    def get_usage(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """Return aggregated ``{model: (input_tokens, output_tokens)}``."""
        return self._backend.get_usage(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
        )

    def get_cached_latency(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> float:
        """Return total latency (seconds) from cached responses."""
        return self._backend.get_cached_latency(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
        )

    def clear(self) -> None:
        """Clear all locally archived records."""
        self._backend.clear()
