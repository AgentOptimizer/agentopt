import asyncio
import logging
import math
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


def test_empty_checkpoint_accepts_extended_limit_on_immediate_stop(tmp_path, monkeypatch):
    options = dict(count=4, checkpoint_path=tmp_path / "empty.db", checkpoint_id="empty")
    stop = Event()
    stop.set()
    seed, _ = test_gittins.make_selector(**options, max_total_question_evaluations=8)
    seed.select_best(stop_event=stop)
    resumed, tracker = test_gittins.make_selector(**options)
    assert resumed.select_best(stop_event=stop).stop_reason == "user_stopped"
    assert not tracker.records
    with sqlite3.connect(options["checkpoint_path"]) as db:
        assert db.execute("SELECT cell_cap FROM metadata").fetchone() == (16,)
        assert db.execute("SELECT COUNT(*) FROM batches").fetchone() == (0,)

    factory = test_gittins.ModelSelector

    def reordered(**kwargs):
        kwargs["model_prices"] = {k: dict(reversed(v.items()))
                                  for k, v in reversed(kwargs["model_prices"].items())}
        return factory(**kwargs)

    monkeypatch.setattr(test_gittins, "ModelSelector", reordered)
    reopened, tracker = test_gittins.make_selector(**options)
    assert reopened.select_best().total_evaluations == len(tracker.records) == 16


def test_unreachable_final_manifest_rejected_without_writes(tmp_path):
    path = tmp_path / "full.db"
    options = dict(count=2, checkpoint_path=path, checkpoint_id="extra")
    seed, _ = test_gittins.make_selector(**options)
    assert seed.select_best().total_evaluations == 8
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO batches SELECT MAX(id)+1, 'solver=cheap', '[0]' FROM batches")
    before = path.read_bytes()
    restored, tracker = test_gittins.make_selector(**options)
    with pytest.raises(CheckpointError, match="all saved batches"):
        restored.select_best()
    assert path.read_bytes() == before
    assert not tracker.records


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize("startup_error", [True, False])
def test_root_handler_reconfiguration_preserves_startup_checks(monkeypatch, when, startup_error):
    # Keep basicConfig(force=True) from closing pytest's own capture handlers.
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    sink = logging.NullHandler()
    factory = logging.getLogRecordFactory()
    running = mitm_runner.DumpMaster.running
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        tracker._backend.open_session("A")

        async def start_b(master):
            await running(master)
            if when == "during":
                logging.basicConfig(force=True, handlers=[sink])
            if startup_error:
                logging.getLogger("embedded.proxy").error("injected B startup error")

        monkeypatch.setattr(mitm_runner.DumpMaster, "running", start_b)
        if when == "before":
            logging.basicConfig(force=True, handlers=[sink])
        if startup_error:
            with pytest.raises(RuntimeError, match="failed to start"):
                tracker._backend.open_session("B")
        else:
            _, port = tracker._backend.open_session("B")
            assert port > 0
    finally:
        tracker.stop()
    assert root.handlers == [sink]
    assert logging.getLogRecordFactory() is factory


def test_factory_replaced_during_startup_survives_cleanup_and_retry(monkeypatch):
    running = mitm_runner.DumpMaster.running
    host_factory = logging.getLogRecordFactory()

    def replacement(*args, **kwargs):
        return host_factory(*args, **kwargs)

    async def reconfigure(master):
        await running(master)
        logging.setLogRecordFactory(replacement)
        logging.error("error after factory replacement")

    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(mitm_runner.DumpMaster, "running", reconfigure)
            with pytest.raises(RuntimeError, match="failed to start"):
                tracker._backend.open_session()
        assert logging.getLogRecordFactory() is replacement
        _, port = tracker._backend.open_session()
        assert port > 0 and logging.getLogRecordFactory() is replacement
    finally:
        logging.setLogRecordFactory(host_factory)
        tracker.stop()


@pytest.mark.parametrize(("budget", "reason"), [
    ({"max_total_question_evaluations": 9}, "question_budget"),
    ({"max_search_cost_usd": 0.0361}, "search_cost_budget"),
])
def test_replay_refuses_batch_outside_original_budget(tmp_path, budget, reason):
    options = dict(count=8, checkpoint_id="budget")
    reference = tmp_path / "reference.db"
    full, _ = test_gittins.make_selector(checkpoint_path=reference, max_total_question_evaluations=10, **options)
    assert full.select_best().total_evaluations == 10
    with sqlite3.connect(reference) as db:
        next_batch = db.execute("SELECT id, arm, questions FROM batches WHERE id=9").fetchone()

    path = tmp_path / "limited.db"
    limited, _ = test_gittins.make_selector(checkpoint_path=path, **budget, **options)
    result = limited.select_best()
    assert result.total_evaluations == 8 and result.stop_reason == reason
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO batches VALUES (?, ?, ?)", next_batch)
    before = path.read_bytes()

    resumed, tracker = test_gittins.make_selector(checkpoint_path=path, **budget, **options)
    updates = []
    with pytest.raises(CheckpointError, match=reason):
        resumed.select_best(on_update=updates.append)
    assert not updates and not tracker.records
    assert path.read_bytes() == before


@pytest.mark.parametrize("emit_error", [False, True])
def test_temporary_host_factory_only_dispatched_errors_block_startup(monkeypatch, emit_error):
    host = logging.getLogRecordFactory()
    running = mitm_runner.DumpMaster.running
    set_factory = logging.setLogRecordFactory
    factory_writes = []
    original_dispatch = logging.Logger.callHandlers

    def set_host_factory(factory):
        factory_writes.append(factory)
        set_factory(factory)

    def temporary(*args, **kwargs):
        return host(*args, **kwargs)

    async def startup(master):
        await running(master)
        set_host_factory(temporary)
        try:
            rebuilt = logging.makeLogRecord({"levelno": logging.INFO, "msg": "host info"})
            logging.getLogger("host").handle(rebuilt)
            record = logging.makeLogRecord({"levelno": logging.ERROR, "msg": "startup failure"})
            if emit_error:
                logging.getLogger("host").handle(record)
        finally:
            set_host_factory(host)

    monkeypatch.setattr(logging, "setLogRecordFactory", set_host_factory)
    monkeypatch.setattr(mitm_runner.DumpMaster, "running", startup)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        if emit_error:
            with pytest.raises(RuntimeError, match="failed to start"):
                tracker._backend.open_session()
        else:
            _, port = tracker._backend.open_session()
            assert port > 0
    finally:
        tracker.stop()
        set_factory(host)
    assert factory_writes == [temporary, host]
    assert logging.Logger.callHandlers is original_dispatch


@pytest.mark.parametrize(("limit", "expected"), [
    (math.nextafter(.038, -math.inf), 8),
    (.038, 10),
    (math.nextafter(.038, math.inf), 10),
])
def test_exact_decimal_batch_budget_and_reopen(tmp_path, limit, expected):
    options = dict(count=8, checkpoint_path=tmp_path / "budget.db", checkpoint_id="boundary",
                   outcomes=dict.fromkeys(["cheap", "middle", "expensive", "dominated"], 0.5),
                   max_search_cost_usd=limit)
    selector, tracker = test_gittins.make_selector(**options)
    result = selector.select_best()
    assert result.total_evaluations == len(tracker.records) == expected
    assert result.selection_cost == (.036 if expected == 8 else .038)
    resumed, tracker = test_gittins.make_selector(**options)
    restored = resumed.select_best()
    assert restored.selection_cost == result.selection_cost
    assert restored.total_evaluations == expected and not tracker.records


def test_saved_charges_resume_at_exact_token_budget(tmp_path, monkeypatch):
    factory = test_gittins.ModelSelector

    def with_decimal_prices(**kwargs):
        for rates in kwargs["model_prices"].values():
            rates["input_price"] = 0.01
        return factory(**kwargs)

    monkeypatch.setattr(test_gittins, "ModelSelector", with_decimal_prices)
    tokens = dict(cheap=1, middle=3, expensive=10, dominated=4)
    path = tmp_path / "decimal.db"
    options = dict(count=8, checkpoint_path=path, checkpoint_id="decimal",
                   tokens=tokens, outcomes=dict.fromkeys(tokens, .5),
                   max_search_cost_usd=3.8e-7)
    fresh, _ = test_gittins.make_selector(**options)
    reference = fresh.select_best()
    assert reference.total_evaluations == 10
    before = path.read_bytes()
    reopened, tracker = test_gittins.make_selector(**options)
    result = reopened.select_best()
    assert not tracker.records and result.total_evaluations == 10
    assert result.selection_cost == reference.selection_cost == 3.8e-7
    assert result.recommendations == reference.recommendations
    assert path.read_bytes() == before


@pytest.mark.parametrize("level", [logging.INFO, logging.ERROR])
def test_prebound_dispatch_of_non_propagating_logger(monkeypatch, level):
    class AddonLogger(logging.Logger):
        callHandlers = logging.Logger.callHandlers

    seen = []

    class Sink(logging.Handler):
        def emit(self, record):
            seen.append(record.levelno)

    addon = AddonLogger("host.addon")
    addon.propagate = False
    addon.addHandler(Sink())
    original_handle = logging.Handler.handle
    running = mitm_runner.DumpMaster.running

    async def startup(master):
        await running(master)
        addon.log(level, "addon startup")

    monkeypatch.setattr(mitm_runner.DumpMaster, "running", startup)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        if level == logging.ERROR:
            with pytest.raises(RuntimeError, match="failed to start"):
                tracker._backend.open_session()
        else:
            _, port = tracker._backend.open_session()
            assert port > 0
    finally:
        tracker.stop()
    assert seen == [level] and logging.Handler.handle is original_handle
