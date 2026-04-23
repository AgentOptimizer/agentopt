"""Global proxy singleton + session management for the ``with router:`` API.

``Router.__enter__`` calls :func:`_activate` which — on first use —
starts a proxy server in the background, installs the httpx redirect,
opens a fresh session port, and binds the router to that session.
``Router.__exit__`` calls :func:`_deactivate` which closes the session
port and detaches the router.  The proxy itself lives until interpreter
shutdown (via ``atexit``) so repeat activations are cheap.

Users never see any of this — the only public surface is
``with router:``.
"""

from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from typing import Any, Optional

from agentopt.proxy.certs import CertificateAuthority
from agentopt.proxy.interceptor import (
    _session_port_var,
    install_redirect,
    uninstall_redirect,
)
from agentopt.proxy.server import ProxyServer


_lock = threading.Lock()
_server: Optional[ProxyServer] = None
_ca: Optional[CertificateAuthority] = None
_redirect_installed: bool = False


@dataclass
class _RoutingHandle:
    """Opaque handle returned by :func:`_activate`."""

    session_id: str
    port: int
    token: Any  # ContextVar reset token


def _ensure_server() -> ProxyServer:
    """Lazy-start the singleton proxy on first call."""
    global _server, _ca, _redirect_installed
    if _server is not None:
        return _server
    with _lock:
        if _server is not None:
            return _server
        _ca = CertificateAuthority()
        server = ProxyServer(ca=_ca)
        server.start()
        install_redirect()
        _redirect_installed = True
        _server = server
        atexit.register(_shutdown)
        return _server


def _shutdown() -> None:
    """Tear down the singleton proxy (registered via ``atexit``)."""
    global _server, _redirect_installed
    if _redirect_installed:
        uninstall_redirect()
        _redirect_installed = False
    if _server is not None:
        _server.stop()
        _server = None


def _activate(router: Any) -> _RoutingHandle:
    """Open a session, bind *router* to it, and route in-process calls."""
    server = _ensure_server()
    session = server.session_manager.create_session(
        data_id=None, combo_id=None, agent_id=None,
    )
    port = server.open_session_port(session.session_id)
    server.set_session_router(session.session_id, router)
    token = _session_port_var.set(port)
    return _RoutingHandle(session_id=session.session_id, port=port, token=token)


def _deactivate(handle: _RoutingHandle) -> None:
    """Reverse of :func:`_activate` — detach router, close session."""
    global _server
    _session_port_var.reset(handle.token)
    server = _server
    if server is None:
        return
    server.clear_session_router(handle.session_id)
    server.close_session_port(handle.session_id)
    server.session_manager.end_session(handle.session_id)
