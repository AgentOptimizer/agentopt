import logging
import sqlite3
import os
from threading import Event, Thread

import pytest

from agentopt import LLMTracker
from agentopt.proxy import mitm_runner
from tests.test_gittins import make_selector


def test_relative_plot_path_survives_cwd_change(tmp_path, monkeypatch):
    from agentopt.visualization import LiveParetoPlot
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    db = tmp_path / "result.png"
    options = dict(count=2, checkpoint_path=db, checkpoint_id="relative")
    selector, _ = make_selector(**options)
    plot = LiveParetoPlot(total_questions=2, output="result", show_window=False)
    try:
        plot.run(selector, on_update=lambda _: os.chdir(tmp_path))
        assert (launch / "result.png").read_bytes().startswith(b"\x89PNG")
        restored, tracker = make_selector(**options)
        restored.select_best()
        assert not tracker.records
    finally:
        plot.plt.close(plot.figure)


@pytest.mark.parametrize("legacy", [False, True])
def test_reordered_custom_prices_restore_same_checkpoint(tmp_path, monkeypatch, legacy):
    from tests import test_gittins
    factory = test_gittins.ModelSelector
    options = dict(count=2, checkpoint_path=tmp_path/"prices.db", checkpoint_id="prices")
    selector, _ = make_selector(**options)
    selector.select_best()
    if legacy:
        with sqlite3.connect(options["checkpoint_path"]) as db:
            db.execute("UPDATE metadata SET identity=? WHERE id=1", (selector._legacy_price_identity,))
        reopened, tracker = make_selector(**options)
        reopened.select_best()
        assert not tracker.records

    def reorder(**kwargs):
        kwargs["model_prices"] = {k: dict(reversed(list(v.items())))
                                  for k, v in reversed(list(kwargs["model_prices"].items()))}
        return factory(**kwargs)

    monkeypatch.setattr(test_gittins, "ModelSelector", reorder)
    restored, tracker = make_selector(**options)
    restored.select_best()
    assert restored.restored_cells == 8 and not tracker.records


def test_final_startup_error_prevents_session_handoff(monkeypatch):
    import asyncio
    original = mitm_runner.DumpMaster.running
    entered, proceed, returned = Event(), Event(), Event()
    async def late_error(master):
        await original(master)
        logging.error("injected late startup error")
        entered.set()
        await asyncio.to_thread(proceed.wait, 5)
    monkeypatch.setattr(mitm_runner.DumpMaster, "running", late_error)
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    errors = []
    def open_session():
        try:
            tracker._backend.open_session()
        except RuntimeError as exc:
            errors.append(exc)
        finally:
            returned.set()
    worker = Thread(target=open_session)
    worker.start()
    try:
        assert entered.wait(10)
        assert not returned.wait(.2)
        proceed.set()
        worker.join(10)
        assert errors and not tracker._backend._masters
    finally:
        proceed.set()
        worker.join(10)
        tracker.stop()

def test_first_draw_of_dense_frontier_has_only_hittable_visible_labels():
    from types import SimpleNamespace
    from matplotlib.backend_bases import MouseEvent
    from agentopt.visualization import LiveParetoPlot
    plot = LiveParetoPlot(total_questions=1, show_window=False)
    rows = [SimpleNamespace(model_name=str(i), combo={"solver": str(i)}, observed_samples=1,
                            conservative_cost_usd=.001 if i == 0 else .002,
                            conservative_accuracy=.3 if i == 0 else .9) for i in range(180)]
    try:
        plot.update(SimpleNamespace(total_evaluations=180, selection_cost=0, stop_reason="running",
                                    is_final=False, get_recommendations=lambda: rows))
        canvas = plot.figure.canvas
        canvas.draw()
        visible = [(artist, name) for _, artist, name in plot._chart_labels if artist.get_visible()]
        assert 0 < len(visible) < len(rows)
        for artist, name in visible:
            x, y = artist.get_window_extent().get_points().mean(axis=0)
            assert plot.axis.bbox.contains(x, y)
            event = MouseEvent("button_release_event", canvas, x, y, button=1)
            assert plot._chart_combination_at(event) == name
    finally:
        plot.plt.close(plot.figure)
