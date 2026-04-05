"""LLMTracker — proxy-based LLM call tracking and session management."""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .cache import ResponseCache
from .interceptor import _session_id_var, install_redirect, uninstall_redirect
from .models import CallRecord
from .server import ProxyServer
from .session import SessionInfo


class LLMTracker:
    """Tracks LLM API calls via a local HTTP proxy server.

    On ``start()`` the tracker launches a lightweight proxy server on
    ``127.0.0.1`` (OS-assigned port) and monkey-patches ``httpx`` so that
    in-process LLM requests are transparently redirected through it.
    Subprocess agents can be directed to the proxy by setting the
    environment variables returned by :meth:`get_session_env`.

    Parameters
    ----------
    cache : bool
        Enable API-level response caching (default ``True``).
    cache_dir : str or Path, optional
        Directory for the SQLite cache database.  Defaults to
        ``"./.agentopt_cache"``.  ``None`` disables disk persistence.

    Usage::

        tracker = LLMTracker()
        tracker.start()

        with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku") as session:
            result = agent(input_data)
            # For subprocess agents:
            # env = {**os.environ, **tracker.get_session_env(session.session_id)}
            # subprocess.run(["my-agent", ...], env=env)

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
        self._server: Optional[ProxyServer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the proxy server and install the httpx redirect.

        Idempotent — calling ``start()`` twice is a no-op.
        """
        if self._active:
            return
        self._server = ProxyServer(cache=self._response_cache)
        self._server.start()
        install_redirect(proxy_base_url=self._server.base_url)
        self._active = True

    def stop(self) -> None:
        """Shut down the proxy, restore httpx, and flush the cache.

        Idempotent.
        """
        if not self._active:
            return

        # Collect records from all (ended + still-active) sessions.
        if self._server is not None:
            with self._lock:
                self._records.extend(self._server.session_manager.get_all_records())
            uninstall_redirect()
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
        """Create a proxy session scoped to this context.

        Yields a :class:`SessionInfo` whose ``session_id`` can be used
        with :meth:`get_session_env` for subprocess agents.
        """
        assert self._server is not None, "call tracker.start() first"
        session = self._server.session_manager.create_session(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id,
        )
        token = _session_id_var.set(session.session_id)
        try:
            yield session
        finally:
            _session_id_var.reset(token)
            self._server.session_manager.end_session(session.session_id)

    @contextmanager
    def track_agent(self, agent_id: str):
        """Update the agent_id on the current session.

        Requires an enclosing :meth:`track` scope.
        """
        session_id = _session_id_var.get()
        old_agent_id: Optional[str] = None
        if session_id and self._server:
            session = self._server.session_manager.get_session(session_id)
            if session is not None:
                old_agent_id = session.agent_id
                session.agent_id = agent_id
        try:
            yield
        finally:
            if session_id and self._server:
                session = self._server.session_manager.get_session(session_id)
                if session is not None:
                    session.agent_id = old_agent_id

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def get_session_env(self, session_id: str) -> Dict[str, str]:
        """Return environment variables for subprocess agents.

        Set these on the subprocess environment so its LLM SDK traffic
        is automatically routed through the proxy::

            env = {**os.environ, **tracker.get_session_env(session.session_id)}
            subprocess.run(["gemini", "cli", ...], env=env)
        """
        assert self._server is not None, "call tracker.start() first"
        url = self._server.session_url(session_id)
        return {
            "OPENAI_BASE_URL": url,
            "ANTHROPIC_BASE_URL": url,
            "GOOGLE_API_BASE": url,
            "AGENTOPT_SESSION_ID": session_id,
            "AGENTOPT_PROXY_URL": self._server.base_url,
        }

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
