"""Remote backend — talks to a long-lived ``agentopt serve`` daemon.

Selected when ``LLMTracker`` is constructed with the
``AGENTOPT_GATEWAY_URL`` env var set.  Surface mirrors :class:`LocalBackend`;
every operation maps to one HTTP call against the daemon's control plane
(see :mod:`.daemon`).

Per-session state held client-side:

* ``proxy_port`` — the daemon-allocated per-session port that
  ``HTTPS_PROXY`` and the in-process :class:`RemoteHandler`'s httpx client
  point at.
* ``ca_bundle_path`` — a per-gateway file combining certifi system CAs
  with the daemon's mitmproxy CA.  Subprocess agents get its path via
  :meth:`get_session_env`; the in-process handler hands it to httpx as
  ``verify=``.

No cache or records live in this process — the daemon owns both.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple

import certifi
import httpx

if TYPE_CHECKING:
    from agentopt.routing.base import Router

from ._backend import _Backend
from .interceptor import (
    ActiveSession,
    CallHandler,
    _active_session_var,
    forward_async,
    forward_sync,
    install_redirect,
    uninstall_redirect,
)
from .models import CallRecord
from .providers import _BUILTIN_PROVIDERS
from .session import SessionInfo

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = httpx.Timeout(300.0)  # generous; LLM calls can be slow


# ---------------------------------------------------------------------------
# CA bundle plumbing — write a per-gateway bundle file once
# ---------------------------------------------------------------------------


def _bundle_dir_for(gateway_url: str) -> Path:
    """Per-gateway directory for the merged CA bundle.

    Keyed by a short hash of the URL so two daemons on different hosts
    keep their bundles separate.
    """
    h = hashlib.sha256(gateway_url.encode()).hexdigest()[:12]
    d = Path.home() / ".agentopt" / "remote" / h
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_ca_bundle(ca_pem: bytes, gateway_url: str) -> str:
    """Combine certifi system CAs + daemon CA, write to the bundle file.

    Idempotent: if the file already contains an identical bundle, no
    write happens.  Returns the absolute path as a string.
    """
    dest = _bundle_dir_for(gateway_url) / "ca-bundle.pem"
    with open(certifi.where(), "rb") as f:
        system_pem = f.read()
    new_bytes = system_pem + b"\n" + ca_pem
    if dest.exists() and dest.read_bytes() == new_bytes:
        return str(dest)
    dest.write_bytes(new_bytes)
    return str(dest)


# ---------------------------------------------------------------------------
# RemoteHandler — in-process httpx → daemon's per-session proxy port
# ---------------------------------------------------------------------------


def _make_httpx_client(
    cls, proxy_url: str, ca_bundle_path: str,
):
    """Construct an httpx ``Client`` or ``AsyncClient`` with proxy config.

    ``proxy=`` is the modern httpx kwarg (0.27+).  Fall back to the
    deprecated ``proxies=`` for older installs (package pin is
    ``httpx>=0.24``).
    """
    try:
        return cls(proxy=proxy_url, verify=ca_bundle_path, timeout=_DEFAULT_TIMEOUT)
    except TypeError:
        return cls(
            proxies={"all://": proxy_url},
            verify=ca_bundle_path,
            timeout=_DEFAULT_TIMEOUT,
        )


class RemoteHandler(CallHandler):
    """Forward intercepted requests through the daemon's session proxy.

    No local cache, no local recording — the daemon does both.  This
    handler exists so that *in-process* Python agents (using httpx via
    OpenAI/Anthropic SDKs) still get their traffic captured when
    ``AGENTOPT_GATEWAY_URL`` is set; subprocess agents use
    ``HTTPS_PROXY`` directly and never hit this code.

    The sync client is created eagerly; the async client is created on
    first use.  This keeps the common (sync-only) path leak-free without
    making the async path more expensive than necessary — and it gives
    :meth:`close` a clear signal for when async cleanup is needed.
    """

    def __init__(self, proxy_url: str, ca_bundle_path: str) -> None:
        self._proxy_url = proxy_url
        self._ca_bundle_path = ca_bundle_path
        self._sync_client: httpx.Client = _make_httpx_client(
            httpx.Client, proxy_url, ca_bundle_path,
        )
        # Lazy: only created if handle_async is ever called.  Avoids
        # leaking a connection pool when only sync paths are exercised.
        self._async_client: Optional[httpx.AsyncClient] = None

    def handle_sync(
        self, client: httpx.Client, request: httpx.Request, *, stream: bool, **kwargs,
    ) -> httpx.Response:
        # Use our proxy-configured client; bypass the patch via forward_sync.
        return forward_sync(self._sync_client, request, stream=stream, **kwargs)

    async def handle_async(
        self,
        client: httpx.AsyncClient,
        request: httpx.Request,
        *,
        stream: bool,
        **kwargs,
    ) -> httpx.Response:
        if self._async_client is None:
            self._async_client = _make_httpx_client(
                httpx.AsyncClient, self._proxy_url, self._ca_bundle_path,
            )
        return await forward_async(
            self._async_client, request, stream=stream, **kwargs,
        )

    def close(self) -> None:
        try:
            self._sync_client.close()
        except Exception:
            logger.debug("ignored exception closing sync client", exc_info=True)
        # Only attempt async cleanup if we actually built an async client.
        # AsyncClient has no synchronous ``close`` — must drive ``aclose``
        # on a loop.
        if self._async_client is not None:
            self._close_async_client(self._async_client)
            self._async_client = None

    @staticmethod
    def _close_async_client(client: httpx.AsyncClient) -> None:
        """Run ``client.aclose()`` regardless of whether a loop is current.

        * No running loop → drive a fresh one via ``asyncio.run``.
        * Running loop (e.g. ``track()`` exits inside ``asyncio.run(...)``)
          → fire-and-forget a task on it; we can't ``await`` from sync
          code, but the task will complete before the loop closes.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            try:
                asyncio.run(client.aclose())
            except Exception:
                logger.debug(
                    "ignored exception draining async client", exc_info=True,
                )
            return

        try:
            loop.create_task(client.aclose())
        except Exception:
            logger.debug(
                "ignored exception scheduling async client aclose", exc_info=True,
            )


# ---------------------------------------------------------------------------
# RemoteBackend
# ---------------------------------------------------------------------------


class _SessionState:
    """Client-side per-session bookkeeping for ``get_session_env`` etc."""

    __slots__ = ("proxy_port", "ca_bundle_path", "handler")

    def __init__(self, proxy_port: int, ca_bundle_path: str, handler: RemoteHandler):
        self.proxy_port = proxy_port
        self.ca_bundle_path = ca_bundle_path
        self.handler = handler


class RemoteBackend(_Backend):
    """Control-plane client for a long-lived ``agentopt serve`` daemon."""

    def __init__(self, gateway_url: str) -> None:
        self._gateway_url = gateway_url.rstrip("/")
        # Extract host for building per-session proxy URLs (control-plane
        # host == proxy host, by construction).
        parsed = httpx.URL(self._gateway_url)
        self._gateway_host = parsed.host
        self._http = httpx.Client(timeout=httpx.Timeout(30.0))

        # Local view of provider path patterns — drives the in-process
        # interceptor's fast-path filter.  Built-ins are pre-loaded;
        # custom providers added via register_provider() are appended and
        # forwarded to the daemon.
        self._path_patterns: Set[str] = set()
        for p in _BUILTIN_PROVIDERS:
            for pattern in p.path_patterns:
                self._path_patterns.add(pattern)

        self._sessions: Dict[str, _SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._active = False

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        if self._active:
            return
        install_redirect()
        # Sanity: daemon reachable.  Fail fast with a clear message.
        try:
            r = self._http.get(f"{self._gateway_url}/health")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            uninstall_redirect()
            raise RuntimeError(
                f"agentopt: gateway at {self._gateway_url} is not reachable. "
                "Is `agentopt serve` running, and is AGENTOPT_GATEWAY_URL "
                f"correct?  ({exc})"
            ) from exc
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        uninstall_redirect()
        # Close any sessions that survived (well-behaved callers exit
        # track() cleanly, but tests / error paths may not).
        with self._sessions_lock:
            ids = list(self._sessions.keys())
        for sid in ids:
            try:
                self.close_session(sid)
            except Exception:
                logger.debug("ignored exception closing session %s", sid, exc_info=True)
        # NOTE: do not close ``self._http`` here.  Callers (``ModelSelector``
        # in particular) invoke ``get_records()`` / ``get_usage()`` *after*
        # ``stop()`` to harvest the run's records — those calls would fail
        # against a closed control-plane client.  Use :meth:`close` for
        # final teardown.
        self._active = False

    def close(self) -> None:
        """Final teardown.  Idempotent; safe to call after ``stop``.

        Releases the control-plane httpx client.  Without this, httpx
        emits an "Unclosed client" warning at GC time, and on PyPy or
        in cycles the close may run much later than expected.
        """
        if self._active:
            self.stop()
        try:
            if not self._http.is_closed:
                self._http.close()
        except Exception:
            logger.debug(
                "ignored exception closing control-plane client", exc_info=True,
            )

    def __del__(self) -> None:
        # Safety net for callers that forget to call ``close()``.  Errors
        # from __del__ are awkward — swallow them rather than print to
        # stderr at interpreter shutdown.
        try:
            self.close()
        except Exception:
            pass

    # -- session management ------------------------------------------

    def open_session(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        router_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SessionInfo, int, str]:
        """POST /sessions; return ``(SessionInfo, proxy_port, ca_bundle_path)``.

        Exposed for parity with :meth:`LocalBackend.open_session` (and
        for testability); :meth:`track` is the usual entry point.

        *router_config* (optional) is the ``{policy, kwargs}`` shape
        :meth:`Router.config` returns; the daemon resolves it server-side
        and attaches the router to this session.  Pass ``None`` to fall
        back to the daemon's default policy (if any).
        """
        assert self._active, "call backend.start() first"
        body_payload: Dict[str, Any] = {
            "data_id": data_id,
            "combo_id": combo_id,
            "agent_id": agent_id,
        }
        if router_config is not None:
            body_payload["router"] = router_config
        resp = self._http.post(f"{self._gateway_url}/sessions", json=body_payload,)
        if resp.status_code == 400:
            # Daemon rejected the router config — surface the error.
            raise RuntimeError(
                f"agentopt daemon rejected the session: "
                f"{resp.json().get('error', resp.text)}"
            )
        resp.raise_for_status()
        body = resp.json()
        session = SessionInfo(
            session_id=body["session_id"],
            data_id=data_id,
            combo_id=combo_id,
            agent_id=agent_id,
        )
        proxy_port = int(body["proxy_port"])
        ca_pem = base64.b64decode(body["ca_pem_b64"])
        bundle_path = _write_ca_bundle(ca_pem, self._gateway_url)
        return session, proxy_port, bundle_path

    def close_session(self, session_id: str) -> None:
        """DELETE /sessions/{id} and drop client-side state."""
        with self._sessions_lock:
            state = self._sessions.pop(session_id, None)
        if state is not None:
            state.handler.close()
        try:
            self._http.delete(f"{self._gateway_url}/sessions/{session_id}")
        except httpx.HTTPError:
            logger.debug(
                "ignored error closing remote session %s", session_id, exc_info=True,
            )

    @contextmanager
    def track(
        self,
        data_id: str,
        combo_id: str,
        agent_id: Optional[str] = None,
        router: Optional["Router"] = None,
    ) -> Iterator[SessionInfo]:
        # Resolve the router (explicit kwarg > with-router ContextVar).
        if router is None:
            from agentopt.routing.base import get_active_router

            router = get_active_router()

        # Serialize the router for the daemon.  ``config()`` raises
        # NotImplementedError for custom routers that don't override
        # ``_config_kwargs()``; surface that as-is so the user knows
        # exactly what's missing.
        router_config: Optional[Dict[str, Any]] = None
        if router is not None:
            router_config = router.config()

        session, proxy_port, bundle_path = self.open_session(
            data_id=data_id,
            combo_id=combo_id,
            agent_id=agent_id,
            router_config=router_config,
        )
        proxy_url = f"http://{self._gateway_host}:{proxy_port}"
        handler = RemoteHandler(proxy_url=proxy_url, ca_bundle_path=bundle_path)
        with self._sessions_lock:
            self._sessions[session.session_id] = _SessionState(
                proxy_port=proxy_port, ca_bundle_path=bundle_path, handler=handler,
            )

        active = ActiveSession(
            session=session,
            port=proxy_port,
            path_patterns=frozenset(self._path_patterns),
            handler=handler,
        )
        token = _active_session_var.set(active)
        try:
            yield session
        finally:
            _active_session_var.reset(token)
            self.close_session(session.session_id)

    def get_session_env(self, session: SessionInfo) -> Dict[str, str]:
        with self._sessions_lock:
            state = self._sessions.get(session.session_id)
        assert state is not None, (
            f"no remote session state for {session.session_id}; "
            "is the session inside its track() scope?"
        )
        proxy = f"http://{self._gateway_host}:{state.proxy_port}"
        return {
            "HTTPS_PROXY": proxy,
            "SSL_CERT_FILE": state.ca_bundle_path,
            "REQUESTS_CA_BUNDLE": state.ca_bundle_path,
            "NODE_EXTRA_CA_CERTS": state.ca_bundle_path,
        }

    # -- provider registry -------------------------------------------

    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        self._path_patterns.update(path_patterns)
        resp = self._http.post(
            f"{self._gateway_url}/providers",
            json={
                "name": name,
                "base_url": base_url,
                "path_patterns": list(path_patterns),
            },
        )
        resp.raise_for_status()

    # -- cache management --------------------------------------------

    def flush_cache(self) -> None:
        self._http.post(f"{self._gateway_url}/cache/flush").raise_for_status()

    def clear_cache(self) -> None:
        self._http.post(f"{self._gateway_url}/cache/clear").raise_for_status()

    # -- record queries ----------------------------------------------

    def _query_params(
        self, data_id: Optional[str], combo_id: Optional[str], agent_id: Optional[str],
    ) -> Dict[str, str]:
        params = {}
        if data_id is not None:
            params["data_id"] = data_id
        if combo_id is not None:
            params["combo_id"] = combo_id
        if agent_id is not None:
            params["agent_id"] = agent_id
        return params

    def get_records(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[CallRecord]:
        resp = self._http.get(
            f"{self._gateway_url}/records",
            params=self._query_params(data_id, combo_id, agent_id),
        )
        resp.raise_for_status()
        return [CallRecord(**d) for d in resp.json()]

    def get_usage(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int]]:
        resp = self._http.get(
            f"{self._gateway_url}/usage",
            params=self._query_params(data_id, combo_id, agent_id),
        )
        resp.raise_for_status()
        return {k: (int(v[0]), int(v[1])) for k, v in resp.json().items()}

    def get_cached_latency(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> float:
        resp = self._http.get(
            f"{self._gateway_url}/cached_latency",
            params=self._query_params(data_id, combo_id, agent_id),
        )
        resp.raise_for_status()
        return float(resp.json()["seconds"])

    def clear(self) -> None:
        """No-op for remote: client holds no record buffer."""
