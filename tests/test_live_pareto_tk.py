"""Opt-in native Tk lifecycle regression; local tracked agents, no model APIs."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("AGENTOPT_TEST_TK") != "1", reason="Requires a desktop; set AGENTOPT_TEST_TK=1")
def test_native_pause_close_resume_finish_close(tmp_path):
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), str(repo), env.get("PYTHONPATH", "")])
    for phase in ("pause-close", "resume-finish-close"):
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                str(tmp_path), phase], capture_output=True, text=True, timeout=90, env=env)
        assert child.returncode == 0, child.stdout + child.stderr
        assert not child.stderr.strip(), child.stderr


@pytest.mark.skipif(os.environ.get("AGENTOPT_TEST_TK") != "1", reason="Requires a desktop; set AGENTOPT_TEST_TK=1")
def test_same_process_window_objects_collected_in_later_worker(tmp_path):
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), str(repo), env.get("PYTHONPATH", "")])
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path), "gc"],
                           capture_output=True, text=True, timeout=90, env=env)
    assert child.returncode == 0, child.stdout + child.stderr
    assert not child.stderr.strip(), child.stderr


def gc_worker():
    import gc
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    from agentopt.visualization import LiveParetoPlot
    from tests.test_gittins import make_selector
    gc.collect()
    gc.disable()
    def first_window():
        plot = LiveParetoPlot(total_questions=2, show_window=True)
        selector, _ = make_selector(count=2)
        plot.run(selector)
        window = plot.figure.canvas.manager.window
        window.after(10, lambda: window.tk.call(window.protocol("WM_DELETE_WINDOW")))
        plot.wait_for_close()
        assert plot._closed
    for _ in range(3):
        first_window()
        def collect_in_worker(question):
            gc.collect()
            return .6
        plot = LiveParetoPlot(total_questions=2, show_window=True)
        selector, _ = make_selector(count=2, outcomes={"cheap": collect_in_worker})
        plot.run(selector)
        window = plot.figure.canvas.manager.window
        window.after(10, lambda: window.tk.call(window.protocol("WM_DELETE_WINDOW")))
        plot.wait_for_close()
    gc.enable()


def worker(directory, phase):
    import json
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    from agentopt.visualization import LiveParetoPlot
    from tests.test_gittins import make_selector

    root = Path(directory)
    selector, tracker = make_selector(checkpoint_path=root / "run.sqlite3", checkpoint_id="tk-close-v1")
    plot = LiveParetoPlot(total_questions=8, warm_total=8, show_window=True,
                          output=root / (phase + ".png"))
    manager = plot.figure.canvas.manager
    errors = []
    manager.window.report_callback_exception = lambda *args: errors.append(str(args))
    original_update = plot.update
    snapshots = []
    requested = False

    def update(snapshot):
        nonlocal requested
        original_update(snapshot)
        snapshots.append(snapshot)
        if not requested and not snapshot.is_final and snapshot.total_evaluations >= (10 if phase == "pause-close" else 14):
            requested = True
            if phase == "pause-close":
                plot.toggle_pause(None)
            else:
                plot.finish_search(None)

    original_pump = plot.pump_events
    closing = False
    def pump():
        nonlocal closing
        original_pump()
        if phase == "pause-close" and plot._pause_active and not closing:
            closing = True
            # Follow the same window-manager close protocol as the title-bar X.
            manager.window.after(10, lambda: manager.window.tk.call(manager.window.protocol("WM_DELETE_WINDOW")))

    plot.update = update
    plot.pump_events = pump
    results = plot.run(selector, max_concurrent=2)
    (root / (phase + ".json")).write_text(results.model_dump_json(), encoding="utf-8")
    if phase == "pause-close":
        assert plot._closed and results.total_evaluations == 10
    else:
        previous = json.loads((root / "pause-close.json").read_text(encoding="utf-8"))
        assert snapshots[0].total_evaluations == selector.restored_cells == 10
        assert snapshots[0].selection_cost == previous["selection_cost"]
        assert results.total_evaluations == 14 and len(tracker.records) == 4
        assert not plot._closed and plot.finished
        manager.window.after(10, lambda: manager.window.tk.call(manager.window.protocol("WM_DELETE_WINDOW")))
        plot.wait_for_close()
        assert plot._closed
    assert not errors, errors
    assert (root / (phase + ".png")).is_file()
    print(phase, "passed", results.total_evaluations, selector.restored_cells, len(tracker.records))


if __name__ == "__main__":
    if sys.argv[2] == "gc":
        gc_worker()
    else:
        worker(sys.argv[1], sys.argv[2])
