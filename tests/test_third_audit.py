"""Regression checks for the public entry points in the third independent review."""
from collections import namedtuple
import logging
import threading

import pytest

from agentopt import LLMTracker
from agentopt.proxy import mitm_runner
from agentopt.model_selection.checkpoint import CheckpointError, fingerprint
from tests.test_gittins import make_selector


def test_sdk_export_does_not_overwrite_checkpoint(tmp_path):
    from agentopt.visualization import LiveParetoPlot
    path = tmp_path / "sdk.png"
    selector, _ = make_selector(checkpoint_path=path, checkpoint_id="sdk")
    selector.select_best(on_update=lambda snapshot: False)
    saved = path.read_bytes()
    selector, tracker = make_selector(checkpoint_path=path, checkpoint_id="sdk")
    plot = LiveParetoPlot(total_questions=8, output=path.with_suffix(""), show_window=False)
    try:
        with pytest.raises(ValueError, match="Path conflict"):
            plot.run(selector)
        assert not tracker.records
        assert path.read_bytes() == saved
    finally:
        plot.plt.close(plot.figure)


def test_identity_and_execution_settings(tmp_path):
    with pytest.raises(ValueError):
        fingerprint(namedtuple("Pair", "a b")(1, 2))
    options = dict(checkpoint_path=tmp_path / "weight.db", checkpoint_id="weight", mode="single")
    first, _ = make_selector(lambda_cost=0, **options)
    first.select_best()
    second, tracker = make_selector(lambda_cost=0.0, **options)
    second.select_best()
    assert not tracker.records
    with pytest.raises(CheckpointError):
        second.batch_size = float(second.batch_size)


def test_application_error_during_startup_is_ignored(monkeypatch):
    run = mitm_runner.DumpMaster.run
    async def with_unrelated_error(self):
        worker = threading.Thread(target=lambda: logging.error("unrelated application failure"))
        worker.start()
        worker.join()
        await run(self)
    monkeypatch.setattr(mitm_runner.DumpMaster, "run", with_unrelated_error)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track():
            pass
    finally:
        tracker.stop()


def test_constructor_error_restores_root_handlers(monkeypatch):
    before = list(logging.getLogger().handlers)
    def fail():
        raise RuntimeError("partial construction")
    monkeypatch.setattr(mitm_runner.addons, "default_addons", fail)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with pytest.raises(RuntimeError, match="partial construction"):
            with tracker.track():
                pytest.fail("unexpected start")
        assert before == list(logging.getLogger().handlers)
        logging.error("post-failure application log")
    finally:
        tracker.stop()


def test_late_construction_cannot_bind_after_timeout(monkeypatch):
    real = mitm_runner.DumpMaster
    release = threading.Event()
    entered = threading.Event()
    ran = []
    class SlowMaster(real):
        def __init__(self, *args, **kwargs):
            entered.set()
            assert release.wait(10)
            super().__init__(*args, **kwargs)
        async def run(self):
            ran.append(True)
            await super().run()
    monkeypatch.setattr(mitm_runner, "DumpMaster", SlowMaster)
    monkeypatch.setattr(mitm_runner, "_STARTUP_TIMEOUT_SECONDS", .4)
    monkeypatch.setattr(mitm_runner, "_SHUTDOWN_TIMEOUT_SECONDS", .1)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with pytest.raises(RuntimeError, match="did not start"):
            tracker._backend.open_session()
        assert entered.is_set()
        owner, = tracker._backend._masters.values()
        with pytest.raises(RuntimeError, match="retry stop"):
            tracker.stop()
    finally:
        release.set()
        monkeypatch.setattr(mitm_runner, "_SHUTDOWN_TIMEOUT_SECONDS", 5)
        tracker.close()
    assert not ran and not owner._port_holder and owner._loop.is_closed()


def test_failed_shutdown_can_be_retried(monkeypatch):
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    session, _ = tracker._backend.open_session()
    owner = tracker._backend._masters[session.session_id]
    def fail():
        raise RuntimeError("injected shutdown failure")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner._master, "shutdown", fail)
            patch.setattr(mitm_runner, "_SHUTDOWN_TIMEOUT_SECONDS", .1)
            with pytest.raises(RuntimeError, match="retry stop"):
                tracker.stop()
            assert tracker._backend._masters[session.session_id] is owner
    finally:
        tracker.close()
    assert owner._loop.is_closed() and not tracker._backend._masters

def test_clear_discards_only_ended_sessions():
    tracker = LLMTracker(cache=False, cache_dir=None)
    manager = tracker._backend._session_manager
    active = manager.create_session()
    ended = manager.create_session()
    live_record, old_record = object(), object()
    manager.add_record(active.session_id, live_record)
    manager.add_record(ended.session_id, old_record)
    manager.end_session(ended.session_id)
    tracker.clear()
    assert manager.get_all_records() == [live_record]


def test_stop_waits_for_concurrent_session_close(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    session, _ = tracker._backend.open_session()
    owner = tracker._backend._masters[session.session_id]
    entered, release = threading.Event(), threading.Event()
    original = owner.stop
    def hold_close():
        entered.set()
        assert release.wait(10)
        original()
    monkeypatch.setattr(owner, "stop", hold_close)
    with ThreadPoolExecutor(2) as pool:
        closing = pool.submit(tracker._backend.close_session, session.session_id)
        assert entered.wait(5)
        stopping = pool.submit(tracker.stop)
        try:
            with pytest.raises(TimeoutError):
                stopping.result(timeout=.1)
        finally:
            release.set()
        closing.result(timeout=10)
        stopping.result(timeout=10)
    assert owner._loop.is_closed()
