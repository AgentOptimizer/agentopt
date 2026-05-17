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
from typing import Dict, Iterator, List, Optional, Tuple, Union

from ._backend import (
    _MITMPROXY_CA_CERT,
    _MITMPROXY_CONFDIR,
    _Backend,
    _ensure_ca_bundle,
)
from ._local_backend import LocalBackend
from .models import CallRecord
from .session import SessionInfo

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

    Usage::

        tracker = LLMTracker()
        tracker.start()

        with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku") as session:
            result = agent.run(input_data)        # in-process: just works
            env = {**os.environ, **tracker.get_session_env(session)}
            subprocess.run(["gemini", "cli", ...], env=env)  # subprocess: just works

        usage = tracker.get_usage(combo_id="gpt4o+haiku")
        tracker.stop()
    """

    _DEFAULT_CACHE_DIR = ".agentopt_cache"
    _GATEWAY_URL_ENV = "AGENTOPT_GATEWAY_URL"

    def __init__(
        self,
        cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = _DEFAULT_CACHE_DIR,
    ) -> None:
        url = os.environ.get(self._GATEWAY_URL_ENV)
        if url:
            # Imported lazily so the local-mode-only install path doesn't
            # have to evaluate the remote backend module.
            from ._remote_backend import RemoteBackend

            self._backend: _Backend = RemoteBackend(gateway_url=url)
        else:
            self._backend = LocalBackend(cache=cache, cache_dir=cache_dir)

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
        """Tear down all live sessions, restore httpx, flush cache."""
        self._backend.stop()

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    @contextmanager
    def track(
        self, data_id: str, combo_id: str, agent_id: Optional[str] = None,
    ) -> Iterator[SessionInfo]:
        """Open a tracking session.  See :meth:`_Backend.track`."""
        with self._backend.track(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
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
