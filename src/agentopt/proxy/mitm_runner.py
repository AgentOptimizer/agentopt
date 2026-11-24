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

* mitmproxy Master, default_addons and the ErrorCheck startup hook.
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

from mitmproxy.options import Options
from mitmproxy import addons
from mitmproxy.master import Master
from mitmproxy.addons.errorcheck import ErrorCheck, ErrorCheckHandler

from ._mitm_context import session_context
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


class _SessionErrorCheck(ErrorCheck):
    name = "errorcheck"

    def __init__(self):
        self.on_startup_complete = None
        self.repeat_errors_on_stderr = False
        self.logger = ErrorCheckHandler()
        owner = threading.get_ident()
        self.logger.addFilter(lambda record: record.thread == owner)
        self.logger.install()

    def finish(self):
        # Master.run calls this only after its final shutdown_if_errored check.
        super().finish()
        if self.on_startup_complete is not None:
            self.on_startup_complete()


class DumpMaster(Master):
    """Embedded regular proxy with explicitly owned startup logging."""
    def __init__(self, opts, *, loop):
        errorcheck = None
        try:
            super().__init__(opts, event_loop=loop, with_termlog=False)
            self.addons.add(*addons.default_addons())
            errorcheck = _SessionErrorCheck()
            self.addons.add(errorcheck)
        except BaseException:
            if errorcheck is not None:
                errorcheck.logger.uninstall()
            legacy = getattr(self, "_legacy_log_events", None)
            if legacy is not None:
                legacy.uninstall()
            raise


class _StartupHook:
    """Tiny addon that captures the bound port once the master is running.

    Kept separate from ``AgentoptAddon`` so the recording addon stays
    focused on flows.  The ``running`` hook fires once per master, after
    ``Proxyserver.listen_addrs`` is populated.
    """

    def __init__(self, port_holder: list, master: DumpMaster) -> None:
        self._master = master
        self._port_holder = port_holder

    def running(self) -> None:
        # ``listen_addrs`` is a method that returns ``list[Address]``
        # where Address is ``(host, port)``.  We pass exactly one mode
        # so the list has exactly one entry.
        # ctx.master is process-global and may belong to another concurrent
        # session. Capture the owner explicitly so ports cannot cross sessions.
        addrs = self._master.addons.get("proxyserver").listen_addrs()
        if addrs:
            self._port_holder.append(addrs[0][1])


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

        self._cancelled = threading.Event()
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
            self._cancelled.set()
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
        self._cancelled.set()
        master = self._master
        if master is not None:
            try:
                master.shutdown()
            except Exception:
                logger.debug("ignored exception in master.shutdown", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                raise RuntimeError("Proxy thread did not exit within the shutdown timeout; retry stop()")
            else:
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
        with session_context():
            self._run_isolated()

    def _run_isolated(self) -> None:
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
            master = DumpMaster(opts, loop=loop)
            self._master = master
            if self._cancelled.is_set():
                return
            master.addons.get("errorcheck").on_startup_complete = self._ready.set
            master.addons.add(
                _StartupHook(self._port_holder, master), self._addon,
            )

            # Drives the loop until shutdown() flips should_exit.
            if not self._cancelled.is_set():
                loop.run_until_complete(master.run())
        except BaseException as exc:
            self._startup_exc = exc
            self._ready.set()  # unblock start()
        finally:
            # Startup can fail before Master.run reaches done(). Its process-wide
            # logging handler must never keep scheduling onto a closed loop.
            if self._master is not None:
                errorcheck = self._master.addons.get("errorcheck")
                cleanups = [self._master._legacy_log_events.uninstall]
                if errorcheck is not None:
                    cleanups.append(errorcheck.logger.uninstall)
                if self._master._termlog_addon is not None:
                    cleanups.append(self._master._termlog_addon.uninstall)
                for cleanup in cleanups:
                    try:
                        cleanup()
                    except Exception:
                        logger.exception("Failed to remove a mitmproxy logging handler")
            try:
                if self._loop is not None and not self._loop.is_closed():
                    # Master.run() can leave accept/connection tasks pending on
                    # Windows. Drain cancellation before closing their loop.
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                    self._loop.run_until_complete(self._loop.shutdown_default_executor(timeout=_SHUTDOWN_TIMEOUT_SECONDS))
                    self._loop.close()
            except Exception:
                logger.exception("Failed to clean up mitmproxy event loop")
            finally:
                self._master = None
                asyncio.set_event_loop(None)
