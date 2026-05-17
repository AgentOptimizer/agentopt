"""Backend abstraction for :class:`LLMTracker`.

``LLMTracker`` delegates every operation to a ``_Backend`` instance.  Two
implementations ship as siblings of this module:

* :class:`._local_backend.LocalBackend` — runs the mitmproxy
  ``SessionMaster`` in the current Python process.
* :class:`._remote_backend.RemoteBackend` — talks to a long-lived
  ``agentopt serve`` daemon over HTTP.

The choice is made in :class:`LLMTracker.__init__` based on whether the
``AGENTOPT_GATEWAY_URL`` environment variable is set.  Both backends
share the same public surface so ``ModelSelector`` (and any other
``LLMTracker`` user) is unaware of which one is active.

This module owns only the abstract surface and CA-bundle helpers shared
by both backends.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import certifi

from .models import CallRecord
from .session import SessionInfo

if TYPE_CHECKING:
    from agentopt.routing.base import Router

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
        """Tear down all sessions and flush state.

        Record queries (``get_records`` / ``get_usage`` / etc.) remain
        valid after ``stop()`` so callers can harvest the run's records;
        :meth:`close` is the final-teardown hook that releases everything
        else (e.g. the remote backend's HTTP client).
        """

    def close(self) -> None:
        """Release every resource the backend holds.  Idempotent.

        Default implementation calls :meth:`stop` if the backend is
        still active.  Subclasses that hold additional resources (long-
        lived HTTP clients, sockets, threads) override to release them.
        """
        if getattr(self, "_active", False):
            self.stop()

    # -- session management ------------------------------------------

    @abstractmethod
    def track(
        self,
        data_id: str,
        combo_id: str,
        agent_id: Optional[str] = None,
        router: Optional["Router"] = None,
    ):
        """Return a context manager that yields a :class:`SessionInfo`.

        Routing (``router``) is library-only in v1; remote backends
        raise ``NotImplementedError`` when a router is supplied.
        """

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
