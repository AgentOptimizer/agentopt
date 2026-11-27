"""Offline regression tests for warm concurrency and the unified UI entry."""
import asyncio
import sqlite3
from threading import Event

import pytest

from tests.test_gittins import make_selector


@pytest.mark.parametrize("show_ui", [False, True])
@pytest.mark.parametrize("parallel,global_cap,warm_cap,dollars,expected", [
    (True, 3, 5, None, 3),
    (True, 6, 2, None, 2),
    (False, 6, 5, None, 1),
    (True, 6, 5, 1.0, 5),
])
def test_peak_warm_cells(monkeypatch, show_ui, parallel, global_cap, warm_cap, dollars, expected):
    selector, _ = make_selector(count=2, max_search_cost_usd=dollars)
    if show_ui:
        from agentopt.visualization import LiveParetoPlot
        initialize = LiveParetoPlot.__init__
        def offscreen(self, **kwargs):
            initialize(self, show_window=False, **kwargs)
        monkeypatch.setattr(LiveParetoPlot, "__init__", offscreen)
    evaluate = selector._evaluate_cell
    active = peak = 0

    async def measured(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.001)
            return await evaluate(*args)
        finally:
            active -= 1

    monkeypatch.setattr(selector, "_evaluate_cell", measured)
    result = selector.select_best(show_ui=show_ui, parallel=parallel, max_concurrent=global_cap,
                                  warm_start_max_concurrent=warm_cap)
    assert peak == expected
    assert result.total_evaluations == 8


def test_failed_warm_cell_resumes_only_missing_cells(tmp_path, monkeypatch):
    path = tmp_path / "warm.db"
    options = dict(checkpoint_path=path, checkpoint_id="warm-parallel")
    selector, _ = make_selector(**options)
    original = selector._evaluate_cell

    async def fail_first(arm, question, agent, parallel):
        if arm == 0 and question == selector._orders[0][0]:
            raise RuntimeError("first warm cell failed")
        return await original(arm, question, agent, parallel)

    monkeypatch.setattr(selector, "_evaluate_cell", fail_first)
    with pytest.raises(RuntimeError, match="first warm cell failed"):
        selector.select_best(parallel=True, max_concurrent=4, warm_start_max_concurrent=4)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT batch FROM cells ORDER BY batch").fetchall() == [(2,), (3,), (4,)]

    stopped, tracker = make_selector(**options)
    stop = Event()
    stop.set()
    partial = stopped.select_best(stop_event=stop)
    assert stopped.restored_cells == partial.total_evaluations == 3
    assert not tracker.records

    resumed, tracker = make_selector(**options)
    result = resumed.select_best(parallel=True, max_concurrent=2, warm_start_max_concurrent=2)
    baseline, _ = make_selector()
    expected = baseline.select_best()
    assert result.total_evaluations == 3 + len(tracker.records) == expected.total_evaluations
    assert result.recommendations == expected.recommendations
    assert result.selection_cost == expected.selection_cost
    assert [[dp.datapoint_index for dp in row.datapoint_results] for row in result] == [
        [dp.datapoint_index for dp in row.datapoint_results] for row in expected]

def test_default_refills_one_slot_before_slow_first_cell_and_waits_for_last(monkeypatch):
    models = {str(i): 0.5 for i in range(6)}
    selector, _ = make_selector(count=4, outcomes=models, tokens=dict.fromkeys(models, 100))
    original = selector._evaluate_cell
    ninth_started = asyncio.Event()
    other_warm_cells_done = asyncio.Event()
    warm_started = warm_done = warm_active = peak = 0
    adaptive_active = adaptive_peak = 0

    async def probe(arm, question, agent, parallel):
        nonlocal warm_started, warm_done, warm_active, peak, adaptive_active, adaptive_peak
        adaptive = selector._calibrated
        if adaptive:
            assert warm_done == 12 and warm_active == 0
            adaptive_active += 1
            adaptive_peak = max(adaptive_peak, adaptive_active)
        else:
            warm_started += 1
            position = warm_started
            warm_active += 1
            peak = max(peak, warm_active)
            if position == 1:
                # A barrier after the first eight would deadlock here.
                await asyncio.wait_for(ninth_started.wait(), 5)
            if position == 9:
                ninth_started.set()
            if position == 12:
                await asyncio.wait_for(other_warm_cells_done.wait(), 5)
        try:
            await asyncio.sleep(0)
            return await original(arm, question, agent, parallel)
        finally:
            if adaptive:
                adaptive_active -= 1
            else:
                warm_done += 1
                warm_active -= 1
                if warm_done == 11:
                    other_warm_cells_done.set()

    monkeypatch.setattr(selector, "_evaluate_cell", probe)
    result = selector.select_best()
    assert result.total_evaluations == 24
    assert (peak, adaptive_peak) == (8, 2)

def test_budgeted_warmup_retries_waiting_cell_after_cheaper_settlement(monkeypatch):
    selector, tracker = make_selector(max_search_cost_usd=.010)
    original = selector._evaluate_cell
    record = tracker.record
    started = []
    third_started, fourth_started = asyncio.Event(), asyncio.Event()
    tokens = {}

    def charge(model, _):
        data_id, _ = tracker.active.get()
        question = int(data_id.rsplit("_", 1)[1]) - 1
        record(model, tokens[model, question])

    async def timed(arm, question, agent, parallel):
        index = len(started) + 1
        started.append((arm, question))
        tokens[agent.model, question] = {1: 3000, 2: 1000}.get(index, 20000)
        if index == 1:
            await asyncio.sleep(0.01)
            assert len(started) == 1  # no price history
        elif index == 2:
            await asyncio.wait_for(third_started.wait(), 3)
            assert len(started) == 3  # .003 spent + .006 reserved
        elif index == 3:
            third_started.set()
            await asyncio.wait_for(fourth_started.wait(), 3)
        elif index == 4:
            fourth_started.set()
        return await original(arm, question, agent, parallel)

    monkeypatch.setattr(tracker, "record", charge)
    monkeypatch.setattr(selector, "_evaluate_cell", timed)
    result = selector.select_best()
    assert result.stop_reason == "warm_start_budget"
    assert result.total_evaluations == len(set(started)) == 4
    assert result.selection_cost == pytest.approx(.044)
    assert result.search_cost_overshoot_usd == pytest.approx(.034)


def test_cap_overshoot_restores_late_settlements_and_missing_warm_cell(tmp_path, monkeypatch):
    options = dict(checkpoint_path=tmp_path / "warm.db", checkpoint_id="priced-warm")
    selector, tracker = make_selector(max_search_cost_usd=.005, **options)
    original = selector._evaluate_cell
    record = tracker.record
    starts = []
    third_settled = asyncio.Event()

    async def fail_one(arm, question, agent, parallel):
        starts.append((arm, question))
        index = len(starts)
        if index == 2:
            await asyncio.wait_for(third_settled.wait(), 3)
            raise RuntimeError("interruption")
        value = await original(arm, question, agent, parallel)
        if index == 3:
            third_settled.set()
        return value

    # The probe costs .001. The other three cells have already been admitted
    # before the higher actual prices become known.
    monkeypatch.setattr(tracker, "record", lambda model, _: record(model, 1000 if not tracker.records else 10000))
    monkeypatch.setattr(selector, "_evaluate_cell", fail_one)
    with pytest.raises(RuntimeError, match="interruption"):
        selector.select_best(max_concurrent=3)
    same, calls = make_selector(max_search_cost_usd=.005, **options)
    limited = same.select_best()
    assert limited.total_evaluations == same.restored_cells == 3
    assert limited.selection_cost == pytest.approx(.021)
    assert not calls.records
    more, calls = make_selector(max_search_cost_usd=1.0, **options)
    complete = more.select_best()
    assert complete.total_evaluations == 3 + len(calls.records) == 32


@pytest.mark.parametrize("show_ui", [True, False])
@pytest.mark.parametrize("cap,expected", [(None, (16, 12)), (5, (5, 5))])
def test_independent_phase_widths(monkeypatch, show_ui, cap, expected):
    models = {str(i): .5 for i in range(8)}
    selector, _ = make_selector(count=14, batch_size=12, outcomes=models,
                                tokens=dict.fromkeys(models, 100))
    if show_ui:
        from agentopt.visualization import LiveParetoPlot
        initialize = LiveParetoPlot.__init__

        def offscreen(plot, **kwargs):
            initialize(plot, show_window=False, **kwargs)

        monkeypatch.setattr(LiveParetoPlot, "__init__", offscreen)
    active = [0, 0]
    peaks = [0, 0]
    evaluate = selector._evaluate_cell

    async def measured(*args):
        phase = int(selector._calibrated)
        active[phase] += 1
        peaks[phase] = max(peaks[phase], active[phase])
        try:
            await asyncio.sleep(.001)
            return await evaluate(*args)
        finally:
            active[phase] -= 1

    monkeypatch.setattr(selector, "_evaluate_cell", measured)
    selector.select_best(show_ui=show_ui, warm_start_max_concurrent=16, max_concurrent=cap)
    assert tuple(peaks) == expected
