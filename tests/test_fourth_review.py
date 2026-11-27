import logging
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


def test_reordered_custom_prices_restore_same_checkpoint(tmp_path, monkeypatch):
    from tests import test_gittins
    factory = test_gittins.ModelSelector
    options = dict(count=2, checkpoint_path=tmp_path/"prices.db", checkpoint_id="prices")
    selector, _ = make_selector(**options)
    selector.select_best()

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


def test_exact_tie_group_starts_at_five_and_excludes_nearby_points():
    from types import SimpleNamespace
    from matplotlib.backend_bases import MouseEvent
    from agentopt.visualization import LiveParetoPlot

    rows = [SimpleNamespace(model_name=str(i), combo={"solver": str(i)}, observed_samples=1,
                            conservative_cost_usd=.001, conservative_accuracy=.5)
            for i in range(6)]
    rows[-1].conservative_accuracy += 1e-12
    plot = LiveParetoPlot(total_questions=1, show_window=False)

    def update(active):
        plot.update(SimpleNamespace(total_evaluations=len(active), selection_cost=0, stop_reason="running",
                                    is_final=False, get_recommendations=lambda: active))
        plot.figure.canvas.draw()

    try:
        four_and_neighbor = rows[:4] + rows[-1:]
        update(four_and_neighbor)
        assert not plot._group_labels
        assert all(label.get_visible() for _, label, _ in plot._chart_labels)
        update(rows)
        group, label = next(iter(plot._group_labels.items()))
        assert group == frozenset(r.model_name for r in rows[:5])
        assert label.get_text() == "5 combinations" and label.get_bbox_patch() is None
        x, y = label.get_window_extent().get_points().mean(axis=0)
        plot._select_chart_combination(MouseEvent("button_release_event", plot.figure.canvas, x, y, button=1))
        assert plot._active_group == group
        assert {name for _, _, name in plot._card_regions} <= group
        update(four_and_neighbor)
        assert plot._active_group is None and not plot._group_labels
    finally:
        plot.plt.close(plot.figure)


def test_nearby_numbers_keep_their_targets_when_view_size_changes():
    from types import SimpleNamespace
    from matplotlib.backend_bases import MouseEvent, ResizeEvent
    from agentopt.visualization import LiveParetoPlot

    plot = LiveParetoPlot(total_questions=10, show_window=False)
    rows = [SimpleNamespace(model_name=str(i), combo={"solver": str(i)}, observed_samples=3,
                            conservative_cost_usd=cost, conservative_accuracy=score)
            for i, (cost, score) in enumerate(((.001, .2), (.002, .60), (.00201, .601)))]
    try:
        plot.update(SimpleNamespace(total_evaluations=9, selection_cost=0, stop_reason="running",
                                    is_final=False, get_recommendations=lambda: rows))
        canvas = plot.figure.canvas
        for inches in (11.8, 15.0, 9.0):
            plot.figure.set_size_inches(inches, 7.2)
            canvas.callbacks.process("resize_event", ResizeEvent("resize_event", canvas))
            canvas.draw()
            limits = plot.axis.get_xlim(), plot.axis.get_ylim()
            for i, (point, label, name) in enumerate(plot._chart_labels):
                assert label.get_visible() and label.get_text() == str(i + 1)
                x, y = label.get_window_extent().get_points().mean(axis=0)
                assert plot._chart_combination_at(MouseEvent("motion_notify_event", canvas, x, y)) == name
                if i:
                    assert plot._chart_leaders[i].get_visible()
                    assert plot._chart_leaders[i].xy == point
                    own = plot.axis.transData.transform(point)
                    other = plot.axis.transData.transform(plot._chart_labels[3-i][0])
                    assert (x-own[0])**2 + (y-own[1])**2 < (x-other[0])**2 + (y-other[1])**2
                canvas.callbacks.process("scroll_event", MouseEvent("scroll_event", canvas, x, y, step=1))
            assert limits == (plot.axis.get_xlim(), plot.axis.get_ylim())
    finally:
        plot.plt.close(plot.figure)


def test_dense_exact_groups_have_distinct_click_targets():
    from types import SimpleNamespace
    from matplotlib.backend_bases import MouseEvent
    from agentopt.visualization import LiveParetoPlot
    rows = []
    def add(count, cost, score):
        for _ in range(count):
            name = str(len(rows))
            rows.append(SimpleNamespace(model_name=name, combo={"solver": name}, observed_samples=1,
                                        conservative_cost_usd=cost, conservative_accuracy=score))
    add(1, .001, .1)
    for index in range(6):
        add(5, .005 + index * .000001, .55 + index * .0001)
    add(1, .01, .9)
    plot = LiveParetoPlot(total_questions=1, show_window=False)
    try:
        plot.update(SimpleNamespace(total_evaluations=len(rows), selection_cost=.1, is_final=False,
                                    stop_reason="running", get_recommendations=lambda: rows))
        canvas = plot.figure.canvas
        canvas.draw()
        assert len(plot._group_labels) == 6
        for group, label in plot._group_labels.items():
            x, y = label.get_window_extent().get_points().mean(axis=0)
            assert plot._chart_combination_at(MouseEvent("button_release_event", canvas, x, y)) == group
    finally:
        plot.plt.close(plot.figure)
