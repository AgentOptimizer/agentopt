"""Embed a mitmproxy ``DumpMaster`` per session, in a background thread.

mitmproxy is built around an asyncio event loop.  We host one Master per
session in a dedicated thread with its own loop; the ``AgentoptAddon`` sees
flows on that thread, our main thread (httpx wrapper, user code) is unaffected.

Lifecycle::

    sm = SessionMaster(host="127.0.0.1", session=..., registry=..., recorder=..., cache=...)
    port = sm.start()        # blocks until the proxy is listening; raises on failure
    # ... subprocess hits 127.0.0.1:port ...
    sm.stop()                # signals shutdown, joins the thread

Embedded API surface we depend on (mitmproxy 12.x — pin in pyproject.toml):

* ``mitmproxy.tools.dump.DumpMaster(options, loop=None, with_termlog=True, with_dumper=True)``
* ``Master.run()`` — coroutine that drives the event loop until ``shutdown()``.
* ``Master.shutdown()`` — sync, thread-safe.  Internally schedules
  ``should_exit.set`` via ``call_soon_threadsafe``.
* Addon hooks: ``running()``, ``request(flow)``, ``response(flow)``,
  ``error(flow)``, ``tls_clienthello(data)``.
* ``Proxyserver.listen_addrs`` — list of ``(host, port)`` tuples bound by
  the proxyserver addon; populated by the time ``running()`` fires.

A major-version bump may break any of these — the pin is intentional.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Optional

from mitmproxy import ctx
from mitmproxy.options import Options
from mitmproxy.tools.dump import DumpMaster

from .cache import ResponseCache
from .mitm_addon import AgentoptAddon
from .providers import ProviderRegistry
from .recording import Recorder
from .session import SessionInfo

if TYPE_CHECKING:
    from agentopt.routing.base import Router

logger = logging.getLogger(__name__)


# Time to wait for the master to bind its port before giving up on start().
_STARTUP_TIMEOUT_SECONDS = 30.0
# Time to wait for the master's loop to exit on stop().
_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class _StartupHook:
    """Tiny addon that captures the bound port once the master is running.

    Kept separate from ``AgentoptAddon`` so the recording addon stays
    focused on flows.  The ``running`` hook fires once per master, after
    ``Proxyserver.listen_addrs`` is populated.
    """

    def __init__(self, ready: threading.Event, port_holder: list) -> None:
        self._ready = ready
        self._port_holder = port_holder

    def running(self) -> None:
        # ``listen_addrs`` is a method that returns ``list[Address]``
        # where Address is ``(host, port)``.  We pass exactly one mode
        # so the list has exactly one entry.
        addrs = ctx.master.addons.get("proxyserver").listen_addrs()
        if addrs:
            self._port_holder.append(addrs[0][1])
        self._ready.set()


class SessionMaster:
    """Per-session mitmproxy DumpMaster running in a background thread."""

    def __init__(
        self,
        session: SessionInfo,
        registry: ProviderRegistry,
        recorder: Recorder,
        cache: Optional[ResponseCache] = None,
        host: str = "127.0.0.1",
        router: Optional["Router"] = None,
    ) -> None:
        self._session = session
        self._registry = registry
        self._recorder = recorder
        self._cache = cache
        self._host = host

        self._addon = AgentoptAddon(session, registry, recorder, cache, router=router)

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._master: Optional[DumpMaster] = None
        self._port: Optional[int] = None

        self._ready = threading.Event()
        self._port_holder: list = []
        self._startup_exc: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Spin up the master and return the bound port.

        Blocks until the proxy is listening or the startup timeout elapses.
        Raises ``RuntimeError`` if the master couldn't start.
        """
        if self._thread is not None:
            assert self._port is not None
            return self._port

        self._thread = threading.Thread(
            target=self._run,
            name=f"agentopt-mitm-{self._session.session_id}",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=_STARTUP_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"mitmproxy session master for {self._session.session_id} "
                f"did not start within {_STARTUP_TIMEOUT_SECONDS}s"
            )
        if self._startup_exc is not None:
            raise RuntimeError(
                f"mitmproxy session master for {self._session.session_id} "
                f"failed to start: {self._startup_exc}"
            ) from self._startup_exc
        if not self._port_holder:
            raise RuntimeError(
                f"mitmproxy session master for {self._session.session_id} "
                f"started but no listen address was captured"
            )
        self._port = self._port_holder[0]
        return self._port

    def stop(self) -> None:
        """Signal the master to exit and join the thread."""
        if self._master is not None:
            try:
                self._master.shutdown()
            except Exception:
                logger.debug("ignored exception in master.shutdown", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            self._thread = None

    @property
    def port(self) -> int:
        """Port the proxy is listening on.  Available after ``start()``."""
        assert self._port is not None, "call start() first"
        return self._port

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            opts = Options(
                mode=[f"regular@{self._host}:0"],
                # Keep mitmproxy's default upstream cert verification —
                # a misconfigured upstream should surface, not be silently
                # accepted.  ``with_termlog=False`` below suppresses the
                # console log noise.
                ssl_insecure=False,
            )
            master = DumpMaster(opts, loop=loop, with_termlog=False, with_dumper=False,)
            master.addons.add(
                _StartupHook(self._ready, self._port_holder), self._addon,
            )
            self._master = master

            # Drives the loop until shutdown() flips should_exit.
            loop.run_until_complete(master.run())
        except BaseException as exc:
            self._startup_exc = exc
            self._ready.set()  # unblock start()
        finally:
            try:
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.close()
            except Exception:
                pass
