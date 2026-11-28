"""Cross-policy regressions for partial results and soft dollar caps."""
import json
from collections import Counter
from threading import Barrier, Event, Lock

import httpx
import pytest

from agentopt import ModelSelector
from agentopt.proxy import LLMTracker


def run_case(method, *, count=6, cap=.02, parallel=False, slots=1, timeout_at=None,
             fail_first=False, checkpoint=None, rendezvous=False, mode="single", batch_size=2):
    calls = []
    questions = []
    lock = Lock()
    arrived = Barrier(2)
    retried = Event()

    def transport(req):
        q = int(json.loads(req.content)["messages"][0]["content"])
        with lock:
            calls.append(q)
            if q not in questions:
                questions.append(q)
            position = questions.index(q) + 1
            attempt = calls.count(q)
        if rendezvous and attempt == 1 and position in (timeout_at, timeout_at + 1):
            arrived.wait(timeout=5)
            if position != timeout_at:
                assert retried.wait(timeout=5)
        if position == timeout_at:
            if attempt <= 2:
                raise httpx.ReadTimeout("mock response lost after dispatch", request=req)
            retried.set()
        return httpx.Response(200, json={"model": "metered",
            "usage": {"prompt_tokens": 100 + q, "completion_tokens": 0}})

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        class Agent:
            def __init__(self, models):
                self.model = models["solver"]

            def run(self, q):
                for attempt in range(3):
                    try:
                        client.post("http://127.0.0.1/v1/chat/completions", json={
                            "model": self.model, "messages": [{"role": "user", "content": str(q)}]})
                        break
                    except httpx.ReadTimeout:
                        if attempt == 2:
                            raise
                if fail_first and q == 0:
                    raise ValueError("first score unavailable")
                return q / count

        options = {"sample_fraction": 1} if method == "bayesian" else {
            "objective_mode": mode, "batch_size": batch_size, "checkpoint_path": checkpoint,
            "checkpoint_id": "unknown-cost-fixture" if checkpoint else None}
        selector = ModelSelector(method=method, agent=Agent, models={"solver": ["metered"]},
            dataset=[(i, None) for i in range(count)], eval_fn=lambda _, actual: actual,
            max_search_cost_usd=cap, tracker=LLMTracker(cache=False),
            model_prices={"metered": {"input_price": 1, "output_price": 0}}, **options)
        result = selector.select_best(parallel=parallel, max_concurrent=slots)
    return result, calls


@pytest.mark.parametrize("parallel", [False, True])
def test_bo_partial_question_identity(parallel):
    pytest.importorskip("botorch")
    result, calls = run_case("bayesian", count=4, fail_first=True, parallel=parallel, slots=2)
    assert sorted(calls) == list(range(4))
    expected = [(q, q / 4, {"metered": 100 + q}) for q in (1, 2, 3)]
    cells = result.results[0].datapoint_results
    assert [(c.datapoint_index, c.score, c.input_tokens) for c in cells] == expected
    payload = json.loads(result.model_dump_json())
    assert [(c["datapoint_index"], c["score"], c["input_tokens"])
            for c in payload["results"][0]["datapoint_results"]] == expected


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("timeout_at", [1, 3])
def test_gittins_stops_and_restores_unknown_charge(tmp_path, mode, timeout_at):
    path = tmp_path / "resume.db"
    options = dict(timeout_at=timeout_at, checkpoint=path, mode=mode)
    result, calls = run_case("gittins", **options)
    assert result.stop_reason == "search_cost_unknown"
    assert result.total_evaluations == timeout_at
    assert sorted(Counter(calls).values()) == [1] * (timeout_at - 1) + [3]
    assert result.unmeasured_failed_attempts == 1
    assert result.selection_cost == pytest.approx(sum(100 + q for q in set(calls)) / 1e6)
    if timeout_at == 3:
        assert result.pending_batch_evaluations == 1
    for cap in (.02, .10):
        restored, calls = run_case("gittins", cap=cap, checkpoint=path, mode=mode)
        assert calls == []
        assert restored.stop_reason == "search_cost_unknown"
        assert restored.total_evaluations == result.total_evaluations
        assert restored.selection_cost == result.selection_cost
        assert restored.unmeasured_failed_attempts == 1


@pytest.mark.parametrize("cap,slots,expected", [(None, 1, 6), (.02, 1, 1), (.02, 2, 2)])
def test_bayesian_unknown_charge_admission(cap, slots, expected):
    pytest.importorskip("botorch")
    result, calls = run_case("bayesian", timeout_at=1, cap=cap, parallel=slots>1,
                             slots=slots, rendezvous=slots>1)
    assert set(calls) == set(range(expected))
    assert len(calls) == expected + 2
    assert result.total_evaluations == expected
    assert result.selection_cost == pytest.approx(sum(100 + q for q in range(expected)) / 1e6)
    assert json.loads(result.model_dump_json())["unmeasured_failed_attempts"] == 1
    if cap is not None:
        assert result.stop_reason == "search_cost_unknown"


def test_gittins_drains_inflight_search_without_refilling_slots(tmp_path):
    options = dict(count=12, mode="multi", batch_size=4, slots=2, parallel=True,
                   checkpoint=tmp_path / "concurrent.db")
    result, calls = run_case("gittins", timeout_at=5, rendezvous=True, **options)
    assert len(set(calls)) == result.total_evaluations == 6
    assert len(calls) == 8
    assert result.stop_reason == "search_cost_unknown"
    assert result.pending_batch_evaluations == 2
    assert result.unmeasured_failed_attempts == 1
    assert result.selection_cost == pytest.approx(sum(100 + q for q in set(calls)) / 1e6)
    restored, calls = run_case("gittins", **options)
    assert not calls
    assert restored.total_evaluations == 6 and restored.pending_batch_evaluations == 2
    assert restored.selection_cost == result.selection_cost
    assert restored.stop_reason == "search_cost_unknown"
