import asyncio
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from unittest.mock import patch

import pytest

from agentopt import LLMTracker
from agentopt.model_selection.checkpoint import CheckpointError
from agentopt.proxy import mitm_runner
from tests import test_gittins


def test_finishing_one_proxy_during_another_startup_error(monkeypatch):
    tracker = LLMTracker(cache=False, cache_dir=None)
    backend = tracker._backend
    first_ready, allow_first, second_logging, allow_second = (Event() for _ in range(4))
    running = mitm_runner.DumpMaster.running

    async def interleave(master):
        await running(master)
        session_id = current_thread().name.removeprefix("agentopt-mitm-")
        if backend._session_manager.get_session(session_id).data_id == "A":
            first_ready.set()
            assert await asyncio.to_thread(allow_first.wait, 10)
            return
        legacy = master._legacy_log_events
        emit = legacy.emit

        def delay_record(record):
            if record.msg == "B startup error":
                second_logging.set()
                assert allow_second.wait(10)
            emit(record)

        with patch.object(legacy, "emit", delay_record):
            logging.error("B startup error")

    monkeypatch.setattr(mitm_runner.DumpMaster, "running", interleave)
    handlers = list(logging.getLogger().handlers)
    tracker.start()
    try:
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(backend.open_session, "A")
            try:
                assert first_ready.wait(10)
                second = pool.submit(backend.open_session, "B")
                assert second_logging.wait(10)
                allow_first.set()
                first.result(timeout=10)
            finally:
                allow_first.set()
                allow_second.set()
            with pytest.raises(RuntimeError, match="failed to start"):
                second.result(timeout=10)
    finally:
        tracker.stop()
    assert logging.getLogger().handlers == handlers


def test_empty_legacy_checkpoint_upgrades_on_immediate_stop(tmp_path, monkeypatch):
    options = dict(count=2, checkpoint_path=tmp_path / "empty.db", checkpoint_id="empty")
    stop = Event()
    stop.set()
    seed, _ = test_gittins.make_selector(**options)
    seed.select_best(stop_event=stop)
    with sqlite3.connect(options["checkpoint_path"]) as db:
        db.execute("UPDATE metadata SET identity=?", (seed._legacy_price_identity,))
    resumed, tracker = test_gittins.make_selector(**options)
    assert resumed.select_best(stop_event=stop).stop_reason == "user_stopped"
    assert not tracker.records
    with sqlite3.connect(options["checkpoint_path"]) as db:
        assert db.execute("SELECT identity FROM metadata").fetchone() == (resumed._checkpoint_identity,)
        assert db.execute("SELECT COUNT(*) FROM batches").fetchone() == (0,)

    factory = test_gittins.ModelSelector

    def reordered(**kwargs):
        kwargs["model_prices"] = {k: dict(reversed(v.items()))
                                  for k, v in reversed(kwargs["model_prices"].items())}
        return factory(**kwargs)

    monkeypatch.setattr(test_gittins, "ModelSelector", reordered)
    reopened, tracker = test_gittins.make_selector(**options)
    assert reopened.select_best().total_evaluations == len(tracker.records) == 8


@pytest.mark.parametrize("legacy", [False, True])
def test_unreachable_final_manifest_rejected_without_writes(tmp_path, legacy):
    path = tmp_path / "full.db"
    options = dict(count=2, checkpoint_path=path, checkpoint_id="extra")
    seed, _ = test_gittins.make_selector(**options)
    assert seed.select_best().total_evaluations == 8
    with sqlite3.connect(path) as db:
        if legacy:
            db.execute("UPDATE metadata SET identity=?", (seed._legacy_price_identity,))
        db.execute("INSERT INTO batches SELECT MAX(id)+1, 'solver=cheap', '[0]' FROM batches")
    before = path.read_bytes()
    restored, tracker = test_gittins.make_selector(**options)
    with pytest.raises(CheckpointError, match="all saved batches"):
        restored.select_best()
    assert path.read_bytes() == before
    assert not tracker.records
