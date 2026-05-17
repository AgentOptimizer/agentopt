"""Backend abstraction for :class:`LLMTracker`.

``LLMTracker`` delegates every operation to a ``_Backend`` instance.  Two
implementations live in the proxy package:

* :class:`LocalBackend` — runs the mitmproxy ``SessionMaster`` in the
  current Python process (the original ``LLMTracker`` behaviour).
* :class:`RemoteBackend` (see :mod:`._remote_backend`) — talks to a
  long-lived ``agentopt serve`` daemon over HTTP.

The choice is made in :class:`LLMTracker.__init__` based on whether the
``AGENTOPT_GATEWAY_URL`` environment variable is set.

The two backends share the same public surface so ``ModelSelector`` (and
any other ``LLMTracker`` user) is unaware of which one is active.
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import certifi

from .cache import ResponseCache
from .interceptor import (
    ActiveSession,
    LocalHandler,
    _active_session_var,
    install_redirect,
    uninstall_redirect,
)
from .mitm_runner import SessionMaster
from .models import CallRecord
from .providers import ProviderRegistry
from .recording import Recorder
from .session import SessionInfo, SessionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CA bundle helpers (shared by both backends — clients always need the bundle
# to verify mitmproxy-impersonated upstream LLM hosts).
# ---------------------------------------------------------------------------

# mitmproxy's confdir defaults to ~/.mitmproxy.  We don't override it —
# users may already have a mitmproxy CA installed in their browser/system
# keychain, and reusing it avoids re-prompting them.
_MITMPROXY_CONFDIR = Path(os.path.expanduser("~/.mitmproxy"))
_MITMPROXY_CA_CERT = _MITMPROXY_CONFDIR / "mitmproxy-ca-cert.pem"
# Where we write the merged bundle (mitmproxy CA + system CAs from certifi).
# Subprocesses set SSL_CERT_FILE to this so they can verify *both*
# mitmproxy-impersonated LLM hosts AND real (passthrough) hosts.
_AGENTOPT_BUNDLE = _MITMPROXY_CONFDIR / "agentopt-bundle.pem"


def _ensure_ca_bundle() -> str:
    """Return path to a CA bundle = mitmproxy CA + certifi system CAs.

    Subprocesses need this — setting ``SSL_CERT_FILE`` to mitmproxy's CA
    alone would break verification for any host the proxy *doesn't* MITM
    (passthrough hosts that the subprocess talks to directly).

    Idempotent: if the bundle is already up to date with mitmproxy's CA,
    no work is done.
    """
    if not _MITMPROXY_CA_CERT.exists():
        raise RuntimeError(
            f"mitmproxy CA cert not found at {_MITMPROXY_CA_CERT}. "
            "Has any SessionMaster started yet?"
        )

    ca_mtime = _MITMPROXY_CA_CERT.stat().st_mtime
    if _AGENTOPT_BUNDLE.exists() and _AGENTOPT_BUNDLE.stat().st_mtime >= ca_mtime:
        return str(_AGENTOPT_BUNDLE)

    with open(certifi.where(), "rb") as f:
        system_pem = f.read()
    with open(_MITMPROXY_CA_CERT, "rb") as f:
        mitm_pem = f.read()

    # Append rather than prepend so existing trust roots take precedence.
    _AGENTOPT_BUNDLE.write_bytes(system_pem + b"\n" + mitm_pem)
    return str(_AGENTOPT_BUNDLE)


# ---------------------------------------------------------------------------
# _Backend ABC
# ---------------------------------------------------------------------------


class _Backend(ABC):
    """Internal interface used by :class:`LLMTracker`.

    Implementations own the proxy lifecycle, session attribution, records,
    cache, and provider registry.  Public methods mirror ``LLMTracker``'s
    surface so the tracker is a thin delegator.
    """

    # -- lifecycle ----------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Prepare the backend for ``track()`` calls."""

    @abstractmethod
    def stop(self) -> None:
        """Tear down all sessions and flush state."""

    # -- session management ------------------------------------------

    @abstractmethod
    def track(
        self, data_id: str, combo_id: str, agent_id: Optional[str] = None,
    ):
        """Return a context manager that yields a :class:`SessionInfo`."""

    @abstractmethod
    def get_session_env(self, session: SessionInfo) -> Dict[str, str]:
        """Env vars that route a subprocess through this session's proxy."""

    # -- provider registry -------------------------------------------

    @abstractmethod
    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        """Add or replace an LLM provider."""

    # -- cache management --------------------------------------------

    @abstractmethod
    def flush_cache(self) -> None:
        """Force-flush any dirty cache entries to disk."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Delete all cached responses."""

    # -- record queries ----------------------------------------------

    @abstractmethod
    def get_records(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[CallRecord]:
        """Return ``CallRecord`` list, filtered by any combination of IDs."""

    @abstractmethod
    def get_usage(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """Return aggregated ``{model: (input_tokens, output_tokens)}``."""

    @abstractmethod
    def get_cached_latency(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> float:
        """Return total latency (seconds) from cached responses."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all locally archived records."""


# ---------------------------------------------------------------------------
# LocalBackend — today's in-process behaviour
# ---------------------------------------------------------------------------


class LocalBackend(_Backend):
    """In-process backend: per-session mitmproxy + httpx monkey-patch.

    This is the original :class:`LLMTracker` implementation, extracted
    behind :class:`_Backend` so a ``RemoteBackend`` can live alongside it
    without forking the tracker's public surface.
    """

    _DEFAULT_CACHE_DIR = ".agentopt_cache"

    def __init__(
        self,
        cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._records: List[CallRecord] = []
        self._lock = threading.Lock()
        self._active = False
        self._response_cache = ResponseCache(cache_dir=cache_dir) if cache else None
        self._session_manager = SessionManager()
        self._registry = ProviderRegistry()
        self._recorder = Recorder(self._session_manager)
        # session_id -> SessionMaster
        self._masters: Dict[str, SessionMaster] = {}
        self._masters_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        """Install the httpx redirect.  Subprocess masters spin up per-track()."""
        if self._active:
            return
        install_redirect()
        self._active = True

    def stop(self) -> None:
        """Tear down all live SessionMasters, restore httpx, flush cache."""
        if not self._active:
            return
        uninstall_redirect()
        # Stop any masters that survived (track() should have closed them,
        # but in test/error paths they may linger).
        with self._masters_lock:
            masters = list(self._masters.values())
            self._masters.clear()
        for m in masters:
            try:
                m.stop()
            except Exception:
                logger.debug("ignored exception in SessionMaster.stop", exc_info=True)
        self._session_manager.force_end_all()
        with self._lock:
            self._records.extend(self._session_manager.get_all_records())
        if self._response_cache is not None:
            self._response_cache.close()
        self._active = False

    # -- session management ------------------------------------------

    def open_session(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Tuple[SessionInfo, int]:
        """Create a session + start its ``SessionMaster``.  No ContextVar.

        Imperative counterpart to :meth:`track`.  Used by the daemon
        (:mod:`.daemon`) where session lifecycle is driven by HTTP
        endpoints, not a context manager.
        """
        assert self._active, "call backend.start() first"
        session = self._session_manager.create_session(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
        )
        master = SessionMaster(
            session=session,
            registry=self._registry,
            recorder=self._recorder,
            cache=self._response_cache,
        )
        try:
            port = master.start()
            with self._masters_lock:
                self._masters[session.session_id] = master
        except Exception:
            try:
                master.stop()
            except Exception:
                logger.exception(
                    "Failed to stop SessionMaster after startup failure for session %s",
                    session.session_id,
                )
            finally:
                self._session_manager.end_session(session.session_id)
            raise
        return session, port

    def close_session(self, session_id: str) -> None:
        """Stop the session's ``SessionMaster`` and archive its records."""
        with self._masters_lock:
            m = self._masters.pop(session_id, None)
        if m is not None:
            m.stop()
        self._session_manager.end_session(session_id)

    @contextmanager
    def track(
        self, data_id: str, combo_id: str, agent_id: Optional[str] = None,
    ) -> Iterator[SessionInfo]:
        """Create a tracking session with its own mitmproxy port.

        Sets the ``_active_session_var`` ContextVar so the in-process
        httpx patch records calls into this session.  Eagerly starts a
        ``SessionMaster`` so subprocesses can use ``get_session_env``
        without further setup.
        """
        session, port = self.open_session(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
        )
        handler = LocalHandler(
            session=session, recorder=self._recorder, cache=self._response_cache,
        )
        active = ActiveSession(
            session=session,
            port=port,
            path_patterns=self._registry.path_patterns,
            handler=handler,
        )
        token = _active_session_var.set(active)

        try:
            yield session
        finally:
            _active_session_var.reset(token)
            self.close_session(session.session_id)

    def get_session_env(self, session: SessionInfo) -> Dict[str, str]:
        """Return env vars that route a subprocess through this session's port."""
        with self._masters_lock:
            master = self._masters.get(session.session_id)
        assert master is not None, (
            f"no SessionMaster for {session.session_id}; "
            "is the session inside its track() scope?"
        )
        bundle = _ensure_ca_bundle()
        return {
            "HTTPS_PROXY": f"http://127.0.0.1:{master.port}",
            "SSL_CERT_FILE": bundle,
            "REQUESTS_CA_BUNDLE": bundle,
            "NODE_EXTRA_CA_CERTS": bundle,
        }

    # -- provider registry -------------------------------------------

    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        """Add or replace a provider.

        The shared ``ProviderRegistry`` is updated for both interception
        paths: the subprocess addon sees the new intercept hostname
        immediately via the shared reference, and the in-process httpx
        patch picks up the new path patterns at the next ``track()``
        entry (which snapshots ``registry.path_patterns`` onto the
        ``ActiveSession``).
        """
        self._registry.register(name, base_url, path_patterns)

    # -- cache management --------------------------------------------

    def flush_cache(self) -> None:
        """Flush dirty cache entries to disk immediately."""
        if self._response_cache is not None:
            self._response_cache.flush()

    def clear_cache(self) -> None:
        """Clear all cached responses and database rows."""
        if self._response_cache is not None:
            self._response_cache.clear()

    # -- record queries ----------------------------------------------

    def _all_records(self) -> List[CallRecord]:
        """Gather records from both the local archive and live sessions."""
        with self._lock:
            records = list(self._records)
        records.extend(self._session_manager.get_all_records())
        return records

    def get_records(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[CallRecord]:
        """Return ``CallRecord`` list, filtered by any combination of IDs."""
        result = []
        for r in self._all_records():
            if data_id is not None and r.data_id != data_id:
                continue
            if combo_id is not None and r.combo_id != combo_id:
                continue
            if agent_id is not None and r.agent_id != agent_id:
                continue
            result.append(r)
        return result

    def get_usage(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """Return aggregated ``{model: (input_tokens, output_tokens)}``."""
        records = self.get_records(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id
        )
        totals: Dict[str, List[int]] = {}
        for r in records:
            if r.model not in totals:
                totals[r.model] = [0, 0]
            totals[r.model][0] += r.prompt_tokens
            totals[r.model][1] += r.completion_tokens
        return {k: (v[0], v[1]) for k, v in totals.items()}

    def get_cached_latency(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> float:
        """Return total latency (seconds) from cached responses."""
        records = self.get_records(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id
        )
        return sum(r.latency_seconds for r in records if r.cached)

    def clear(self) -> None:
        """Clear all locally archived records."""
        with self._lock:
            self._records.clear()
