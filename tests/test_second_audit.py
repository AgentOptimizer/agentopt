"""Independent-review regressions: local HTTP only, no model APIs."""
import concurrent.futures
import http.client
import json
import logging
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentopt import LLMTracker
from agentopt.model_selection.checkpoint import CheckpointError, EvaluationCheckpoint, fingerprint
from tests.test_gittins import make_selector
from tests.test_benchmark_runner import runner


def test_suffixless_plot_collision_rejected_before_evaluation(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"input":"q","expected":"a"}\n', encoding="utf-8")
    cp = tmp_path / "figure.png"
    selector, _ = make_selector(checkpoint_path=cp, checkpoint_id="old")
    selector.select_best(on_update=lambda s: s.total_evaluations < 3)
    original = cp.read_bytes()
    with pytest.raises(SystemExit) as error:
        runner.main(["--dataset", str(data), "--models", "cheap", "--objective-mode", "multi",
                     "--plot-output", str(tmp_path / "figure"), "--checkpoint", str(cp),
                     "--output", str(tmp_path / "report.json")])
    assert error.value.code == 2
    assert cp.read_bytes() == original


@pytest.mark.parametrize("damage", ["DELETE FROM metadata", "DROP TABLE metadata", "DROP TABLE cells", "PRAGMA user_version=0"])
def test_missing_checkpoint_structure_not_recreated(tmp_path, damage):
    cp = tmp_path / "saved.db"
    selector, _ = make_selector(checkpoint_path=cp, checkpoint_id="old")
    selector.select_best(on_update=lambda s: s.total_evaluations < 3)
    with sqlite3.connect(cp) as db:
        db.execute(damage)
    original = cp.read_bytes()
    selector, tracker = make_selector(checkpoint_path=cp, checkpoint_id="NEW")
    with pytest.raises(CheckpointError):
        selector.select_best()
    assert not tracker.records
    assert cp.read_bytes() == original


def test_hardlinked_checkpoint_rejected(tmp_path):
    cp = tmp_path / "saved.db"
    with EvaluationCheckpoint(cp, "identity"):
        pass
    alias = tmp_path / "alias.db"
    alias.hardlink_to(cp)
    for name in (cp, alias):
        with pytest.raises(CheckpointError, match="hard link"):
            with EvaluationCheckpoint(name, "identity"):
                pytest.fail("hardlinked file acquired")


def test_fingerprint_preserves_observable_python_types():
    assert fingerprint((1, 2)) != fingerprint([1, 2])
    assert fingerprint({1: "a"}) != fingerprint({"1": "a"})
    assert fingerprint({"b": 2, "a": 1}) != fingerprint({"a": 1, "b": 2})
    assert fingerprint(True) != fingerprint(1)


@pytest.mark.parametrize("setting,value", [("batch_size", 1), ("lambda_cost", 2), ("lambda_latency", 2)])
def test_settings_immutable_before_and_during_checkpoint_run(tmp_path, setting, value):
    cp = tmp_path / "saved.db"
    selector, tracker = make_selector(checkpoint_path=cp, checkpoint_id="stable")
    with pytest.raises(CheckpointError, match="immutable"):
        setattr(selector, setting, value)
    def change(snapshot):
        if snapshot.total_evaluations == 8:
            setattr(selector, setting, value)
    with pytest.raises(CheckpointError, match="immutable"):
        selector.select_best(on_update=change)
    resumed, fresh = make_selector(checkpoint_path=cp, checkpoint_id="stable")
    result = resumed.select_best(on_update=lambda s: False)
    assert result.total_evaluations == 8
    assert not fresh.records


def test_wire_concurrent_sessions_and_stopped_records(mock_upstream):
    for _ in range(3):
        barrier = threading.Barrier(3, timeout=20)
        def worker(number):
            tracker = LLMTracker(cache=False, cache_dir=None)
            tracker.register_provider("local", mock_upstream.base_url, ("/v1/chat/completions",))
            tracker.start()
            try:
                with tracker.track(data_id=str(number)) as session:
                    port = tracker._backend._masters[session.session_id].port
                    barrier.wait()
                    client = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    try:
                        client.request("POST", mock_upstream.base_url + "/v1/chat/completions",
                            body=json.dumps({"model":"gpt-4o-mini", "messages":[{"role":"user", "content":str(number)}]}),
                            headers={"Content-Type":"application/json"})
                        response = client.getresponse()
                        response.read()
                        assert response.status == 200
                    finally:
                        client.close()
                    barrier.wait()
                    records = tracker.get_records()
                    assert [(r.data_id, r.request_body["messages"][0]["content"]) for r in records] == [(str(number), str(number))]
                    barrier.wait()
                before = tracker.get_records()
            finally:
                tracker.stop()
            assert tracker.get_records() == before
            tracker.start()
            tracker.stop()
            assert tracker.get_records() == before
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            list(pool.map(worker, range(3)))


def test_early_addon_failure_releases_logging_handlers(monkeypatch):
    from mitmproxy.addonmanager import AddonManager
    from agentopt.proxy.mitm_runner import _StartupHook
    original = AddonManager.add
    handlers = list(logging.getLogger().handlers)
    def fail(self, *addons):
        if any(isinstance(addon, _StartupHook) for addon in addons):
            raise RuntimeError("early registration failure")
        return original(self, *addons)
    monkeypatch.setattr(AddonManager, "add", fail)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with pytest.raises(RuntimeError, match="early registration failure"):
            with tracker.track():
                pytest.fail("started")
    finally:
        tracker.stop()
    assert list(logging.getLogger().handlers) == handlers
    logging.getLogger(__name__).error("ordinary log after simulated startup failure")


def test_stop_waits_for_start_and_does_not_hand_out_stopped_session(monkeypatch):
    from agentopt.proxy.mitm_runner import SessionMaster
    bound, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = SessionMaster.start
    def delayed(self):
        port = original(self)
        bound.set()
        assert release.wait(15)
        return port
    monkeypatch.setattr(SessionMaster, "start", delayed)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    backend = tracker._backend
    errors, results = [], []
    def open_session():
        try:
            results.append(backend.open_session("test"))
        except RuntimeError as exc:
            errors.append(str(exc))
    opening = threading.Thread(target=open_session)
    opening.start()
    assert bound.wait(15)
    closing = threading.Thread(target=lambda: (tracker.stop(), stopped.set()))
    closing.start()
    try:
        # Wait for the backend to close its acceptance gate, not arbitrary sleep.
        with backend._lifecycle:
            assert backend._lifecycle.wait_for(lambda: not backend._active, timeout=5)
        assert not stopped.is_set()
    finally:
        release.set()
        opening.join(15)
        closing.join(15)
    assert not opening.is_alive() and not closing.is_alive()
    assert errors and not results and not backend._masters
    assert not [t for t in threading.enumerate() if t.name.startswith("agentopt-mitm-")]


def test_edge_labels_stay_inside_axes_and_select_correct_cards():
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backend_bases import MouseEvent
    from agentopt.visualization import LiveParetoPlot
    values = {"low": .4, **{f"tie-{i}": .9 for i in range(7)}}
    selector, _ = make_selector(count=2, outcomes=values, tokens={m: 1000 if m == "low" else 2000 for m in values})
    plot = LiveParetoPlot(total_questions=2, show_window=False)
    plot.update(selector.select_best())
    try:
        expected_group = frozenset(name for _, _, name in plot._chart_labels[1:])
        assert set(plot._group_labels) == {expected_group}
        for expected in (plot._chart_labels[0][2], expected_group):
            plot.figure.canvas.draw()
            label = (plot._group_labels[expected_group] if expected == expected_group
                     else plot._chart_labels[0][1])
            box = label.get_window_extent()
            assert plot.axis.bbox.contains(box.x0, box.y0)
            assert plot.axis.bbox.contains(box.x1, box.y1)
            event = MouseEvent("button_release_event", plot.figure.canvas,
                               (box.x0+box.x1)/2, (box.y0+box.y1)/2, button=1)
            plot._select_chart_combination(event)
            if expected == expected_group:
                assert plot._active_group == expected_group
            else:
                assert plot._selected_combination == expected
    finally:
        plot.plt.close(plot.figure)


def test_session_creation_failure_does_not_block_stop(monkeypatch):
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    def fail(**kwargs):
        raise RuntimeError("creation failed")
    monkeypatch.setattr(tracker._backend._session_manager, "create_session", fail)
    with pytest.raises(RuntimeError, match="creation failed"):
        tracker._backend.open_session()
    assert tracker._backend._starting == 0
    tracker.stop()
