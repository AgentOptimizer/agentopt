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
        result = selector.select_best(parallel=parallel, warm_start_max_concurrent=1, on_update=update)
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


def test_budget_increases_are_cumulative_and_monotone(tmp_path):
    path = tmp_path / "extend.db"
    calls = []

    def outcome(model, score):
        def run(question):
            calls.append((model, question))
            return score
        return run

    outcomes = {m: outcome(m, s) for m, s in
                {"cheap": .6, "middle": .8, "expensive": .9, "dominated": .5}.items()}

    def make(**limits):
        return make_selector(checkpoint_path=path, checkpoint_id="same-workflow", outcomes=outcomes, **limits)

    first, _ = make(max_search_cost_usd=.001, observation_budget_fraction=.5)
    partial = first.select_best()
    assert partial.stop_reason == "warm_start_budget"
    bigger, tracker = make(max_search_cost_usd=.5, observation_budget_fraction=.75)
    result = bigger.select_best()
    assert result.total_evaluations == 24
    assert len(calls) == len(set(calls)) == 24
    assert len(tracker.records) == 24 - partial.total_evaluations
    resumed_cells = cells(result)
    assert all(resumed_cells[key] == dp for key, dp in cells(partial).items())
    original_bytes = path.read_bytes()
    for change in (dict(max_total_question_evaluations=22), dict(max_search_cost_usd=.49),
                   dict(seed=43), dict(checkpoint_id="changed")):
        options = dict(checkpoint_path=path, checkpoint_id="same-workflow", max_search_cost_usd=.5)
        rejected, tracker = make_selector(**(options | change))
        with pytest.raises(ValueError, match="only increase|Checkpoint setup differs"):
            rejected.select_best()
        assert not tracker.records and path.read_bytes() == original_bytes


@pytest.mark.parametrize("version", [1, 2, 3])
def test_old_checkpoint_requires_fresh_path(tmp_path, version):
    path = tmp_path / "old.db"
    selector, tracker = make_selector(checkpoint_path=path, checkpoint_id="same")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY, identity TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES (1, ?)", (selector._identity_for_limits(selector.cell_cap, None),))
        db.execute("CREATE TABLE batches (id INTEGER PRIMARY KEY, arm TEXT NOT NULL, questions TEXT NOT NULL)")
        db.execute("CREATE TABLE cells (batch INTEGER NOT NULL, question INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (batch, question))")
        db.execute(f"PRAGMA user_version={version}")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Use a new path"):
        selector.select_best()
    assert not tracker.records and path.read_bytes() == before


def test_concurrent_run_is_rejected_and_lock_released(tmp_path):
    path = tmp_path / "lock.sqlite3"
    with EvaluationCheckpoint(path, lambda *_: "identity", limits=(1, None)):
        with pytest.raises(RuntimeError, match="already in use"):
            with EvaluationCheckpoint(path, lambda *_: "identity", limits=(1, None)):
                pytest.fail("Must not acquire a second lock")
    with EvaluationCheckpoint(path, lambda *_: "identity", limits=(1, None)) as journal:
        assert journal.saved_cells == 0


def test_hard_process_exit_preserves_cell_inside_incomplete_batch(tmp_path):
    path = tmp_path / "interrupted.sqlite3"
    script = r'''
import os, sys
from tests.test_gittins import make_selector
from agentopt.model_selection.checkpoint import EvaluationCheckpoint
original = EvaluationCheckpoint.save_cell_at
count = 0
def save(self, batch, question, data):
    global count
    original(self, batch, question, data)
    count += 1
    if count == 9:  # Eight warm cells plus one cell of a two-cell batch.
        os._exit(23)
EvaluationCheckpoint.save_cell_at = save
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


class DefaultAgent:
    def __init__(self, models):
        self.model = models["answer"]

    def run(self, question):
        default_tracker.record(self.model, 1000)
        return 1


def default_score(expected, actual):
    return float(expected == actual)


def test_agent_property_resumes_on_budget_extension(tmp_path, monkeypatch):
    from agentopt import ModelSelector
    from tests.test_gittins import RecordingTracker

    class Agent(DefaultAgent):
        @property
        def model_name(self):
            return self.model

    monkeypatch.chdir(tmp_path)
    for cap, restored in [(4, 0), (8, 4)]:
        tracker = RecordingTracker()
        monkeypatch.setattr(sys.modules[__name__], "default_tracker", tracker, raising=False)
        selector = ModelSelector(agent=Agent, eval_fn=default_score, models={"answer": ["test"]},
                                 dataset=[(i, 1) for i in range(8)], tracker=tracker,
                                 model_prices={"test": {"input_price": 1, "output_price": 0}},
                                 max_total_question_evaluations=cap)
        assert selector.select_best().total_evaluations == cap
        assert selector.restored_cells == restored and len(tracker.records) == 4


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_public_default_checkpoint(monkeypatch, tmp_path, mode):
    from agentopt import ModelSelector
    from tests.test_gittins import RecordingTracker
    monkeypatch.chdir(tmp_path)

    def selector(**options):
        tracker = RecordingTracker()
        monkeypatch.setattr(sys.modules[__name__], "default_tracker", tracker, raising=False)
        return ModelSelector(agent=DefaultAgent, eval_fn=default_score, models={"answer": ["test"]},
                             dataset=[(i, 1) for i in range(8)], tracker=tracker, objective_mode=mode,
                             model_prices={"test": {"input_price": 1, "output_price": 0}},
                             **options), tracker

    first, _ = selector(max_total_question_evaluations=4)
    initial = first.select_best()
    assert first.checkpoint_path.is_file()
    second, calls = selector(max_total_question_evaluations=8)
    assert second.checkpoint_path == first.checkpoint_path
    result = second.select_best()
    assert second.restored_cells == initial.total_evaluations == 4
    assert result.total_evaluations == 8 and len(calls.records) == 4
    assert result.selection_cost == pytest.approx(0.008)
    different, _ = selector(seed=8)
    assert different.checkpoint_path != first.checkpoint_path
    fresh, calls = selector(checkpoint_path=None)
    fresh.select_best()
    assert fresh.checkpoint_path is None and len(calls.records) == 8


def test_default_window_exports_and_resumes(monkeypatch, tmp_path):
    import json
    from PIL import Image
    from agentopt import ModelSelector
    from agentopt.visualization import LiveParetoPlot
    from tests.test_gittins import RecordingTracker
    monkeypatch.chdir(tmp_path)
    initialize = LiveParetoPlot.__init__
    def offscreen(plot, **options):
        initialize(plot, show_window=False, **options)
    monkeypatch.setattr(LiveParetoPlot, "__init__", offscreen)
    for expected_new in (4, 0):
        tracker = RecordingTracker()
        monkeypatch.setattr(sys.modules[__name__], "default_tracker", tracker, raising=False)
        selector = ModelSelector(agent=DefaultAgent, eval_fn=default_score, objective_mode="multi",
                                 models={"answer": ["test"]}, dataset=[(i, 1) for i in range(4)],
                                 tracker=tracker, model_prices={"test": {"input_price": 1, "output_price": 0}})
        result = selector.select_best(show_ui=True)
        assert len(tracker.records) == expected_new
        assert selector.restored_cells + expected_new == result.total_evaluations == 4
    exports = list((tmp_path / ".agentopt_cache" / "exports").glob("*/results.json"))
    assert len(exports) == 2
    for path in exports:
        assert json.loads(path.read_text("utf-8"))["total_evaluations"] == 4
        with Image.open(path.with_name("frontier.png")) as image:
            assert image.format == "PNG"


def test_automatic_identity_distinguishes_closures_and_adjacent_scorers():
    from agentopt.model_selection.checkpoint import workflow_identity

    def scorer(weight):
        return lambda expected, actual: weight * (expected == actual)

    left, right = (lambda x, y: x), (lambda x, y: y)
    assert workflow_identity(DefaultAgent, left) != workflow_identity(DefaultAgent, right)
    assert workflow_identity(DefaultAgent, scorer(2)) == workflow_identity(DefaultAgent, scorer(2))
    assert workflow_identity(DefaultAgent, scorer(2)) != workflow_identity(DefaultAgent, scorer(3))


def test_runtime_workflow_can_resume_with_reported_version(monkeypatch, tmp_path):
    import re
    from agentopt import ModelSelector
    from tests.test_gittins import RecordingTracker
    monkeypatch.chdir(tmp_path)

    def create(**options):
        tracker = RecordingTracker()

        class Agent:
            def __init__(self, models):
                self.model = models["answer"]

            def run(self, question):
                tracker.record(self.model, 1000)
                return 1

        return ModelSelector(agent=Agent, eval_fn=default_score, models={"answer": ["test"]},
                             dataset=[(0, 1)], tracker=tracker,
                             model_prices={"test": {"input_price": 1, "output_price": 0}}, **options), tracker

    with pytest.warns(RuntimeWarning, match="stable workflow identity") as emitted:
        original, _ = create()
    version = re.search(r"checkpoint_id='([^']+)'", str(emitted[0].message)).group(1)
    original.select_best()
    resumed, tracker = create(checkpoint_id=version)
    assert resumed.checkpoint_path == original.checkpoint_path
    resumed.select_best()
    assert resumed.restored_cells == 1 and len(tracker.records) == 0


def test_failed_charge_survives_budget_extension_and_reopen(tmp_path):
    checkpoint = tmp_path / "failed.db"
    broken = True

    def fail_once(question):
        nonlocal broken
        if broken:
            broken = False
            raise ValueError("model call was already recorded")
        return 1

    def run(limit):
        selector, tracker = make_selector(
            count=6, outcomes={"only": fail_once}, tokens={"only": 1000},
            max_search_cost_usd=limit, checkpoint_path=checkpoint, checkpoint_id="attempts")
        return selector.select_best(parallel=False), tracker.records

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        run(.001)
    for limit, new_calls, total_cells, spent in [
            (.001, 0, 0, .001), (.003, 2, 2, .003), (.003, 0, 2, .003), (.02, 4, 6, .007)]:
        result, records = run(limit)
        assert len(records) == new_calls
        assert result.total_evaluations == total_cells
        assert result.selection_cost == pytest.approx(spent)
        assert result.unmeasured_failed_attempts == 0


def test_failed_adaptive_batch_restores_observations_before_checking_retry_budget(tmp_path):
    calls = []

    def fail_fourth(question):
        calls.append(question)
        if len(calls) == 4:
            raise ValueError("failed after being charged")
        return .5

    def run(limit):
        selector, tracker = make_selector(
            count=6, outcomes={"only": fail_fourth}, tokens={"only": 1000},
            max_search_cost_usd=limit, checkpoint_path=tmp_path / "adaptive.db", checkpoint_id="attempts")
        return selector.select_best(parallel=False), tracker.records

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        run(.004)
    result, records = run(.004)
    assert not records and result.total_evaluations == 3 and result.pending_batch_evaluations == 1
    assert result.selection_cost == pytest.approx(.004)
    result, records = run(.01)
    assert len(records) == 3 and result.total_evaluations == 6
    assert result.selection_cost == pytest.approx(.007)


def test_factory_class_attributes_version_the_automatic_checkpoint():
    from agentopt.model_selection.checkpoint import workflow_identity
    def agent_factory(value):
        class Agent:
            config = {"score": value}
            def run(self, question):
                return self.config["score"]
        return Agent
    assert workflow_identity(agent_factory(.2), default_score) == workflow_identity(agent_factory(.2), default_score)
    assert workflow_identity(agent_factory(.2), default_score) != workflow_identity(agent_factory(.9), default_score)


def test_changed_class_setting_after_construction_refuses_old_checkpoint(tmp_path, monkeypatch):
    from agentopt import ModelSelector
    from tests.test_gittins import RecordingTracker
    monkeypatch.chdir(tmp_path)
    class Agent(DefaultAgent):
        setting = .2
    tracker = RecordingTracker()
    selector = ModelSelector(agent=Agent, eval_fn=default_score, models={"answer": ["test"]},
                             dataset=[(0, 1)], tracker=tracker)
    Agent.setting = .9
    with pytest.raises(ValueError, match="Workflow changed after construction"):
        selector.select_best()
    assert not tracker.records and not selector.checkpoint_path.exists()


def test_parallel_failed_attempts_each_count_once(tmp_path):
    from threading import Lock
    from tests.test_gittins import RecordingTracker
    tracker = RecordingTracker()
    lock = Lock()
    attempts = 0

    def fail_first_adaptive_batch(question):
        nonlocal attempts
        with lock:
            attempts += 1
            if attempts in (3, 4):
                raise ValueError("scoring failed after charge")
        return .5

    def run():
        selector, _ = make_selector(
            tracker=tracker, count=6, outcomes={"only": fail_first_adaptive_batch},
            tokens={"only": 1000}, checkpoint_path=tmp_path / "parallel.db", checkpoint_id="v1")
        return selector.select_best(parallel=True, max_concurrent=2)

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        run()
    assert attempts == 4
    for _ in range(2):
        result = run()
        assert result.total_evaluations == 6
        assert attempts == len(tracker.records) == 8
        assert result.selection_cost == pytest.approx(.008)


def test_partial_batch_known_spend_blocks_new_dispatch_until_budget_extension(tmp_path, monkeypatch):
    checkpoint = tmp_path / "partial.sqlite3"
    calls = []
    def run(limit):
        selector, tracker = make_selector(count=6, batch_size=2, outcomes={"paid": .6}, tokens={"paid": 1000},
                                         max_search_cost_usd=limit, checkpoint_id="partial",
                                         checkpoint_path=checkpoint)
        record = tracker.record
        def charge(model, _):
            calls.append(tracker.active.get())
            record(model, 1000 if len(calls) <= 2 else 10000)
        tracker.record = charge
        return selector.select_best(parallel=False)

    original = EvaluationCheckpoint.save_cell_at
    def interrupt(journal, batch, question, data):
        original(journal, batch, question, data)
        if len(calls) == 3:
            raise InterruptedError("after commit")
    with monkeypatch.context() as patch:
        patch.setattr(EvaluationCheckpoint, "save_cell_at", interrupt)
        with pytest.raises(InterruptedError, match="after commit"):
            run(.004)
    for _ in range(2):
        result = run(.004)
        assert result.stop_reason == "search_cost_budget"
        assert result.total_evaluations == len(calls) == 3
        assert result.pending_batch_evaluations == 1
        assert result.selection_cost == pytest.approx(.012)
    result = run(.1)
    assert result.total_evaluations == len(calls) == 6
    assert result.selection_cost == pytest.approx(.042)


def test_indirect_score_dependency_and_mutual_recursion():
    from types import ModuleType
    from agentopt.model_selection.checkpoint import workflow_identity
    module = ModuleType("metrics")
    exec("def first(n): return second(n-1) if n else 1\n"
         "def second(n): return first(n-1) if n else 1", module.__dict__)
    namespace = {"metrics": module, "first": module.first}
    exec("def score(a, b): return metrics.first(a) + first(b)", namespace)
    def identity():
        return workflow_identity(DefaultAgent, namespace["score"])
    before = identity()
    assert identity() == before
    exec("def second(n): return first(n-1) if n else 0", module.__dict__)
    assert identity() != before
