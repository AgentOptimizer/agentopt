"""Regression cases from the independent checkpoint/UI audit; no model APIs."""
import json
import sqlite3
from threading import Event

import pytest

from agentopt.model_selection.checkpoint import CheckpointError, EvaluationCheckpoint
from tests.test_gittins import make_selector
from tests.test_benchmark_runner import runner


@pytest.mark.parametrize("roles", [("dataset", "output"), ("output", "plot_output"),
                                  ("partial", "checkpoint"), ("partial", "plot_output"),
                                  ("dataset", "checkpoint_lock")])
def test_output_role_collisions_rejected_before_selection(tmp_path, roles):
    paths = {"dataset": tmp_path / "data.jsonl", "output": tmp_path / "report.json",
             "plot_output": tmp_path / "plot.png", "checkpoint": tmp_path / "run.db"}
    paths["partial"] = tmp_path / "report.partial.json"
    paths["checkpoint_lock"] = tmp_path / "run.db.lock"
    a, b = roles
    if b == "checkpoint_lock":
        paths[a] = paths[b]
    else:
        paths[b] = paths[a]
    paths["dataset"].write_text('{"input":"q","expected":"a"}\n', encoding="utf-8")
    before = paths["dataset"].read_bytes()
    args = ["--models", "cheap", "--objective-mode", "multi"]
    for name in ("dataset", "output", "plot_output", "checkpoint"):
        args += ["--" + name.replace("_", "-"), str(paths[name])]
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2
    assert paths["dataset"].read_bytes() == before


def test_existing_hardlink_alias_rejected(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text("original", encoding="utf-8")
    output = tmp_path / "result.json"
    output.hardlink_to(dataset)
    args = runner.make_parser().parse_args(["--dataset", str(dataset), "--models", "cheap", "--output", str(output)])
    with pytest.raises(ValueError, match="Path conflict"):
        runner.validate_output_paths(args)


def partial_checkpoint(path, monkeypatch):
    original = EvaluationCheckpoint.save_cell
    def interrupt(self, q, data):
        if self.db.execute("SELECT COUNT(*) FROM cells").fetchone()[0] == 9:
            raise OSError("simulated interruption")
        original(self, q, data)
    with monkeypatch.context() as patch:
        patch.setattr(EvaluationCheckpoint, "save_cell", interrupt)
        selector, _ = make_selector(checkpoint_path=path, checkpoint_id="audit")
        with pytest.raises(OSError):
            selector.select_best()


@pytest.mark.parametrize("pre_stopped", [True, False])
def test_partial_restore_publishes_saved_cells_before_new_requests(tmp_path, monkeypatch, pre_stopped):
    path = tmp_path / "partial.db"
    partial_checkpoint(path, monkeypatch)
    selector, tracker = make_selector(checkpoint_path=path, checkpoint_id="audit")
    stop = Event()
    if pre_stopped:
        stop.set()
    updates = []
    def update(snapshot):
        updates.append(snapshot)
        assert not tracker.records
        return False
    result = selector.select_best(stop_event=stop, on_update=update)
    assert result.total_evaluations == 9 and result.stop_reason == "user_stopped"
    assert result.pending_batch_evaluations == 1
    assert updates[0].total_evaluations == 9
    resumed, new = make_selector(checkpoint_path=path, checkpoint_id="audit")
    completed = resumed.select_best(on_update=lambda s: s.total_evaluations < 14)
    baseline, _ = make_selector()
    expected = baseline.select_best(on_update=lambda s: s.total_evaluations < 14)
    assert len(new.records) == 5
    assert completed.recommendations == expected.recommendations
    assert completed.selection_cost == expected.selection_cost


def test_changed_public_dataset_rejected(tmp_path):
    path = tmp_path / "dataset.db"
    selector, tracker = make_selector(checkpoint_path=path, checkpoint_id="dataset")
    selector.dataset[0] = (999, None)
    with pytest.raises(CheckpointError, match="inputs changed"):
        selector.select_best()
    assert not tracker.records and not path.exists()


def test_dataset_changes_in_callback_do_not_change_active_run(tmp_path):
    selector, tracker = make_selector(checkpoint_path=tmp_path / "run.db", checkpoint_id="dataset")
    def update(snapshot):
        selector.dataset[:] = []
    result = selector.select_best(on_update=update)
    assert result.total_evaluations == len(tracker.records) == 32


@pytest.mark.parametrize("field,value", [("charge", -17), ("cost", float("inf")),
                                        ("score", 2), ("datapoint_index", 999),
                                        ("latency_seconds", -1), ("input_tokens", {"cheap": -1})])
def test_logically_corrupt_cell_rejected_without_calls(tmp_path, field, value):
    path = tmp_path / "invalid.db"
    selector, _ = make_selector(count=2, checkpoint_path=path, checkpoint_id="invalid")
    selector.select_best()
    with sqlite3.connect(path) as db:
        payload = json.loads(db.execute("SELECT data FROM cells WHERE batch=1").fetchone()[0])
        target = payload if field in ("charge", "cost") else payload["datapoint"]
        target[field] = value
        db.execute("UPDATE cells SET data=? WHERE batch=1", (json.dumps(payload),))
    before = path.read_bytes()
    selector, tracker = make_selector(count=2, checkpoint_path=path, checkpoint_id="invalid")
    with pytest.raises(CheckpointError, match="Invalid checkpoint contents"):
        selector.select_best()
    assert not tracker.records and path.read_bytes() == before


def test_checkpoint_io_errors_identify_operation_and_path(tmp_path):
    path = tmp_path / "broken.db"
    path.write_bytes(b"not a database")
    selector, tracker = make_selector(checkpoint_path=path, checkpoint_id="io")
    with pytest.raises(CheckpointError) as error:
        selector.select_best()
    assert str(path) in runner.failure_message(error.value)
    assert "open/read checkpoint" in runner.failure_message(error.value)
    assert not tracker.records and path.read_bytes() == b"not a database"
    path = tmp_path / "write.db"
    selector, _ = make_selector(checkpoint_path=path, checkpoint_id="io")
    selector.select_best(on_update=lambda s: s.total_evaluations < 8)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TRIGGER reject_cell BEFORE INSERT ON cells BEGIN SELECT RAISE(FAIL, 'test disk failure'); END")
    selector, _ = make_selector(checkpoint_path=path, checkpoint_id="io")
    with pytest.raises(CheckpointError) as error:
        selector.select_best()
    assert "write checkpoint" in runner.failure_message(error.value)
    assert str(path) in str(error.value) and isinstance(error.value.__cause__, sqlite3.Error)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM cells").fetchone()[0] == 8


def test_restore_warm_boundary_does_not_clear_recommendations(tmp_path):
    path = tmp_path / "warm.db"
    selector, _ = make_selector(checkpoint_path=path, checkpoint_id="warm")
    original = selector.select_best(on_update=lambda s: not bool(s.recommendations))
    assert original.total_evaluations == 8 and original.recommendations
    selector, tracker = make_selector(checkpoint_path=path, checkpoint_id="warm")
    snapshots = []
    def update(snapshot):
        snapshots.append(snapshot)
        return False
    selector.select_best(on_update=update)
    assert snapshots[0].total_evaluations == 8
    assert snapshots[0].recommendations == original.recommendations
    assert not tracker.records


def test_coincident_recommendations_have_separate_labels_and_cycle():
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backend_bases import MouseEvent
    from agentopt.visualization import LiveParetoPlot
    selector, _ = make_selector(count=2, outcomes={"cheap": .8, "middle": .8}, tokens={"cheap": 1000, "middle": 1000})
    result = selector.select_best()
    assert len(result.recommendations) == 2
    plot = LiveParetoPlot(total_questions=2, show_window=False)
    plot.update(result)
    plot.figure.canvas.draw()
    labels = [label for _, label, _ in plot._chart_labels]
    assert not labels[0].get_window_extent().overlaps(labels[1].get_window_extent())
    for index in range(2):
        _, label, name = plot._chart_labels[index]
        # Select each current label using its screen location.
        bounds = label.get_window_extent()
        event = MouseEvent("button_release_event", plot.figure.canvas, (bounds.x0+bounds.x1)/2, (bounds.y0+bounds.y1)/2, button=1)
        plot._select_chart_combination(event)
        assert plot._selected_combination == name
        plot.figure.canvas.draw()
    point = plot._chart_labels[0][0]
    x, y = plot.axis.transData.transform(point)
    selected = []
    for _ in range(2):
        event = MouseEvent("button_release_event", plot.figure.canvas, x, y, button=1)
        plot._select_chart_combination(event)
        selected.append(plot._selected_combination)
        plot.figure.canvas.draw()
    assert len(set(selected)) == 2
    card, _, name = plot._card_regions[1]
    bounds = card.get_window_extent()
    x, y = (bounds.x0+bounds.x1)/2, (bounds.y0+bounds.y1)/2
    for action in ("button_press_event", "button_release_event"):
        plot.figure.canvas.callbacks.process(action, MouseEvent(action, plot.figure.canvas, x, y, button=1))
    assert plot._selected_combination == name
    plot.plt.close(plot.figure)
