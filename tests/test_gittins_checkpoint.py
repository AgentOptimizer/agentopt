"""Checkpoint recovery uses local agents only; no provider requests."""

import os
import sqlite3
import subprocess
import sys

import pytest

from agentopt.model_selection.checkpoint import EvaluationCheckpoint
from tests.test_gittins import make_selector


def cells(result):
    return {(row.model_name, dp.datapoint_index): dp
            for row in result.results for dp in row.datapoint_results}


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("parallel", [False, True])
def test_resume_warm_and_adaptive_without_repeating_cells(tmp_path, mode, parallel):
    path = tmp_path / "run.sqlite3"

    def run(limit, checkpoint=path):
        selector, tracker = make_selector(mode=mode, checkpoint_path=checkpoint, checkpoint_id="workflow-v1")
        snapshots = []
        def update(snapshot):
            snapshots.append(snapshot)
            return snapshot.total_evaluations < limit
        result = selector.select_best(parallel=parallel, on_update=update)
        return result, tracker, snapshots, selector

    warm, first, _, _ = run(3)
    adapt, second, updates, selector = run(12)
    assert selector.restored_cells == 3
    assert updates[0].total_evaluations == 3
    assert len(first.records) == 3 and len(second.records) == 9
    resumed, third, updates, _ = run(18)
    baseline, _, _, _ = run(18, tmp_path / "continuous.sqlite3")
    assert len(third.records) == 6 and updates[0].total_evaluations == 12
    assert all(cells(resumed)[key] == dp for key, dp in cells(adapt).items())
    assert cells(resumed).keys() == cells(baseline).keys()
    assert resumed.recommendations == baseline.recommendations
    assert resumed.selection_cost == baseline.selection_cost
    assert set(cells(warm)).issubset(cells(resumed))


@pytest.mark.parametrize("change", [
    {"seed": 43}, {"mode": "single"}, {"count": 9},
    {"checkpoint_id": "workflow-v2"}, {"eta_decay": .25},
    {"score_bounds": (0, 2)},
])
def test_reject_incompatible_setup_before_agent_calls(tmp_path, change):
    path = tmp_path / "run.sqlite3"
    kwargs = dict(checkpoint_path=path, checkpoint_id="workflow-v1")
    first, _ = make_selector(**kwargs)
    first.select_best(on_update=lambda s: s.total_evaluations < 3)
    second, tracker = make_selector(**(kwargs | change))
    with pytest.raises(ValueError, match="Checkpoint setup differs"):
        second.select_best()
    assert tracker.records == [] and tracker.stopped


def test_completed_checkpoint_does_not_construct_agent(tmp_path):
    kwargs = dict(checkpoint_path=tmp_path / "done.sqlite3", checkpoint_id="workflow-v1", count=2)
    first, _ = make_selector(**kwargs)
    original = first.select_best()
    second, tracker = make_selector(**kwargs)
    def forbidden(*_):
        raise AssertionError("Saved cells must not construct an agent")
    second.agent = forbidden
    restored = second.select_best()
    assert restored.total_evaluations == 8
    assert cells(restored) == cells(original)
    assert restored.selection_cost == original.selection_cost
    assert restored.recommendations == original.recommendations
    assert tracker.records == []


def test_concurrent_run_is_rejected_and_lock_released(tmp_path):
    path = tmp_path / "lock.sqlite3"
    with EvaluationCheckpoint(path, "identity"):
        with pytest.raises(RuntimeError, match="already in use"):
            with EvaluationCheckpoint(path, "identity"):
                pytest.fail("Must not acquire a second lock")
    with EvaluationCheckpoint(path, "identity") as journal:
        assert journal.saved_cells == 0


def test_hard_process_exit_preserves_cell_inside_incomplete_batch(tmp_path):
    path = tmp_path / "interrupted.sqlite3"
    script = r'''
import os, sys
from tests.test_gittins import make_selector
from agentopt.model_selection.checkpoint import EvaluationCheckpoint
original = EvaluationCheckpoint.save_cell
count = 0
def save(self, question, data):
    global count
    original(self, question, data)
    count += 1
    if count == 9:  # Eight warm cells plus one cell of a two-cell batch.
        os._exit(23)
EvaluationCheckpoint.save_cell = save
selector, _ = make_selector(checkpoint_path=sys.argv[1], checkpoint_id="workflow-v1")
selector.select_best()
'''
    worker = subprocess.run([sys.executable, "-c", script, str(path)],
                            env=os.environ.copy(), timeout=60, capture_output=True, text=True)
    assert worker.returncode == 23, worker.stderr
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM cells").fetchone()[0] == 9
    selector, tracker = make_selector(checkpoint_path=path, checkpoint_id="workflow-v1")
    result = selector.select_best(on_update=lambda s: s.total_evaluations < 14)
    baseline, _ = make_selector()
    expected = baseline.select_best(on_update=lambda s: s.total_evaluations < 14)
    assert selector.restored_cells == 9
    assert len(tracker.records) == 5
    assert result.total_evaluations == 14
    assert cells(result).keys() == cells(expected).keys()
    assert result.recommendations == expected.recommendations
    assert result.selection_cost == expected.selection_cost


def test_checkpoint_requires_explicit_workflow_version(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_id is required"):
        make_selector(checkpoint_path=tmp_path / "run.sqlite3")


def test_close_and_reopen_view_restores_progress_and_spend(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.backend_bases import CloseEvent
    from agentopt.visualization import LiveParetoPlot

    path = tmp_path / "window.sqlite3"
    first, _ = make_selector(checkpoint_path=path, checkpoint_id="workflow-v1")
    plot = LiveParetoPlot(total_questions=8, warm_total=8, show_window=False)
    def close(snapshot):
        if snapshot.total_evaluations >= 12 and not snapshot.is_final:
            plot.figure.canvas.callbacks.process("close_event", CloseEvent("close_event", plot.figure.canvas))
    stopped = plot.run(first, on_update=close)
    assert stopped.stop_reason == "user_stopped"
    plot.plt.close(plot.figure)

    resumed, tracker = make_selector(checkpoint_path=path, checkpoint_id="workflow-v1")
    reopened = LiveParetoPlot(total_questions=8, warm_total=8, show_window=False)
    snapshots = []
    def finish(snapshot):
        snapshots.append(snapshot)
        if snapshot.total_evaluations >= 16 and not snapshot.is_final:
            reopened.finish_search(None)
    result = reopened.run(resumed, on_update=finish)
    assert snapshots[0].total_evaluations == stopped.total_evaluations == 12
    assert snapshots[0].selection_cost == stopped.selection_cost
    assert snapshots[0].recommendations == stopped.recommendations
    assert len(tracker.records) == 4
    assert reopened._last_evaluations == result.total_evaluations == 16
    assert reopened.metric_values[0].get_text() == "50.00%"
    assert reopened.metric_values[1].get_text() == f"${result.selection_cost:.4f}"
    reopened.plt.close(reopened.figure)
