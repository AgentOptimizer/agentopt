"""LLMTracker — proxy-based LLM call tracking and session management."""

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .cache import ResponseCache
from .certs import CertificateAuthority
from .interceptor import (
    _session_port_var,
    install_redirect,
    register_extra_llm_paths,
    uninstall_redirect,
)
from .models import CallRecord
from .server import ProxyServer
from .session import SessionInfo


class LLMTracker:
    """Tracks LLM API calls via a local HTTP proxy server.

    Each :meth:`track` session gets its own TCP port.  The port **is** the
    session identity — no session IDs in URLs or ContextVars needed.

    * **In-process agents** — the httpx monkey-patch rewrites requests to
      ``http://127.0.0.1:{session_port}/...`` (plaintext HTTP, bypasses
      ``HTTPS_PROXY``).
    * **Subprocess agents** — inherit ``HTTPS_PROXY`` pointing at the same
      session port → CONNECT-mode TLS termination.

    One mechanism, one port per session, both agent types.

    Usage::

        tracker = LLMTracker()
        tracker.start()

        with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku"):
            result = agent.run(input_data)        # in-process: just works
            subprocess.run(["gemini", "cli", ...]) # subprocess: just works

        usage = tracker.get_usage(combo_id="gpt4o+haiku")
        tracker.stop()
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
        self._ca = CertificateAuthority()
        self._server: Optional[ProxyServer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the proxy server and install the httpx redirect."""
        if self._active:
            return
        self._server = ProxyServer(cache=self._response_cache, ca=self._ca)
        self._server.start()
        install_redirect()
        self._active = True

    def stop(self) -> None:
        """Shut down the proxy, restore httpx, and flush the cache."""
        if not self._active:
            return
        if self._server is not None:
            uninstall_redirect()
            self._server.session_manager.force_end_all()
            with self._lock:
                self._records.extend(self._server.session_manager.get_all_records())
            self._server.stop()
            self._server = None
        if self._response_cache is not None:
            self._response_cache.close()
        self._active = False

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    @contextmanager
    def track(
        self, data_id: str, combo_id: str, agent_id: Optional[str] = None,
    ):
        """Create a tracking session with its own proxy port.

        Sets the ``_session_port_var`` ContextVar so the httpx monkey-patch
        routes in-process LLM calls to the session port.  Does NOT mutate
        ``os.environ`` — subprocess agents should pass
        :meth:`get_session_env` explicitly::

            with tracker.track(...) as session:
                env = {**os.environ, **tracker.get_session_env(session)}
                subprocess.run(["tb", "run", ...], env=env)

        This keeps parallel evaluation fully safe — each subprocess gets
        its own env dict pointing at its own port.
        """
        assert self._server is not None, "call tracker.start() first"
        session = self._server.session_manager.create_session(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
        )

        # Bind a port for this session.
        session_port = self._server.open_session_port(session.session_id)

        # Set ContextVar for in-process httpx monkey-patch.
        token = _session_port_var.set(session_port)

        try:
            yield session
        finally:
            _session_port_var.reset(token)
            self._server.close_session_port(session.session_id)
            self._server.session_manager.end_session(session.session_id)

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def get_session_env(self, session: SessionInfo) -> Dict[str, str]:
        """Return env vars that route a subprocess through this session's port.

        Usage::

            with tracker.track(data_id="dp_1", combo_id="c") as session:
                env = {**os.environ, **tracker.get_session_env(session)}
                subprocess.run(["tb", "run", ...], env=env)

        Each session has its own port, so parallel subprocess evaluation
        is fully safe — each subprocess gets its own ``HTTPS_PROXY``.
        """
        assert self._server is not None, "call tracker.start() first"
        port = self._server.get_session_port(session.session_id)
        ca_bundle = self._ca.ca_bundle_path
        return {
            "HTTPS_PROXY": f"http://127.0.0.1:{port}",
            "SSL_CERT_FILE": ca_bundle,
            "REQUESTS_CA_BUNDLE": ca_bundle,
            "NODE_EXTRA_CA_CERTS": ca_bundle,
        }

    # ------------------------------------------------------------------
    # Provider registry
    # ------------------------------------------------------------------

    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        """Add or replace a provider in the proxy's registry.

        Updates three places at once so the new provider works for both
        agent shapes:
        * the proxy's per-instance registry (direct-mode resolution)
        * the proxy's CONNECT-mode intercept-host set (subprocess agents)
        * the in-process httpx patch's path-pattern set, so direct-mode
          calls to the new provider get redirected at all.
        """
        assert self._server is not None, "call tracker.start() first"
        self._server.register_provider(name, base_url, path_patterns)
        register_extra_llm_paths(path_patterns)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def flush_cache(self) -> None:
        """Flush dirty cache entries to disk immediately."""
        if self._response_cache is not None:
            self._response_cache.flush()

    def clear_cache(self) -> None:
        """Clear all cached responses and database rows."""
        if self._response_cache is not None:
            self._response_cache.clear()

    # ------------------------------------------------------------------
    # Record queries
    # ------------------------------------------------------------------

    def _all_records(self) -> List[CallRecord]:
        """Gather records from both the server and the local archive."""
        with self._lock:
            records = list(self._records)
        if self._server is not None:
            records.extend(self._server.session_manager.get_all_records())
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
        """Clear all recorded data."""
        with self._lock:
            self._records.clear()
