"""Local HTTP and failed-start cleanup regressions (no external endpoints)."""
import os
from pathlib import Path
import subprocess
import sys


def test_repeated_local_http_sessions_leave_no_pending_tasks():
    script = r"""
import asyncio, gc, logging, threading
from tests.conftest import MockUpstream
from tests.test_gittins import test_real_tracker_http_calls_produce_costs_and_recommendations
for mode in ['single', 'multi'] * 3:
    server = MockUpstream()
    server.start()
    try:
        test_real_tracker_http_calls_produce_costs_and_recommendations(server, mode)
    finally:
        thread = server._thread
        server.stop()
    assert not thread.is_alive()
    assert server._loop.is_closed()
    assert not [t for t in threading.enumerate() if t.name.startswith('agentopt-mitm-')]
    for handler in logging.getLogger().handlers:
        master = getattr(handler, 'master', None)
        assert master is None or not master.event_loop.is_closed()
    gc.collect()
print('six local HTTP sessions passed')
"""
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    env['PYTHONPATH'] = os.pathsep.join([str(repo / 'src'), str(repo), env.get('PYTHONPATH', '')])
    child = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True,
                           text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    assert not child.stderr.strip(), child.stderr


def test_failed_start_uninstalls_its_logging_handler(monkeypatch):
    import asyncio
    import logging
    import pytest
    from agentopt.proxy.mitm_runner import DumpMaster, SessionMaster
    from agentopt.proxy.providers import ProviderRegistry
    from agentopt.proxy.recording import Recorder
    from agentopt.proxy.session import SessionInfo, SessionManager
    async def failed_start(self):
        asyncio.create_task(asyncio.sleep(3600), name='failed-start-pending-task')
        raise RuntimeError('simulated startup failure')
    monkeypatch.setattr(DumpMaster, 'run', failed_start)
    handlers = list(logging.getLogger().handlers)
    master = SessionMaster(SessionInfo('lifecycle-test'), ProviderRegistry(), Recorder(SessionManager()))
    with pytest.raises(RuntimeError, match='failed to start'):
        master.start()
    master.stop()
    assert master._thread is None
    assert master._loop.is_closed()
    assert list(logging.getLogger().handlers) == handlers
