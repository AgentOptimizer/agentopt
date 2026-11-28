"""Single-objective UCB-E: original matrix coordinates and per-attempt billing."""
import asyncio
from collections import Counter
import json

import httpx
import pytest

from agentopt import ModelSelector
from agentopt.proxy import LLMTracker


def exercise(tmp_path, *, count, slots=1, fraction=1, limit=None, timeouts=0,
             retries=0, fail_zero=False, pre_send=False, synchronize=False):
    requests = []
    arrivals = Counter()
    tracker = LLMTracker(cache=False)

    ready = asyncio.Event()
    timed_out = asyncio.Event()

    async def transport(request):
        if pre_send:
            raise httpx.ConnectTimeout("before reaching upstream", request=request)
        return await upstream(request)

    async def upstream(request):
        question = int(json.loads(request.content)["messages"][0]["content"])
        arrivals[question] += 1
        event = {"question": question, "attempt": arrivals[question]}
        requests.append(event)
        if synchronize:
            if len(arrivals) == slots:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=5)
        if question == first_question and arrivals[question] <= timeouts:
            event["outcome"] = "read_timeout"
            timed_out.set()
            raise httpx.ReadTimeout("response lost after arrival", request=request)
        if synchronize and question != first_question:
            await asyncio.wait_for(timed_out.wait(), timeout=5)
        # All successful billing scenarios charge exactly $0.0001 per call.
        tokens = 100 if not fail_zero and fraction == 1 else 100 + question
        completion = 0 if tokens == 100 else question
        score = question / max(1, count - 1)
        event.update(outcome="response", score=score, input_tokens=tokens, output_tokens=completion)
        return httpx.Response(200, json={"model": "metered", "score": score,
            "usage": {"prompt_tokens": tokens, "completion_tokens": completion}})

    # With seed=42 the first selected question is 3 on both larger grids.
    first_question = 0 if count == 1 else 3
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport),
                               headers={"Authorization": "Bearer offline-test"})
    class Agent:
        def __init__(self, models):
            self.model = models["solver"]

        async def run(self, question):
            for retry in range(retries + 1):
                try:
                    response = await client.post("http://127.0.0.1/v1/chat/completions",
                        json={"model": self.model, "messages": [{"role":"user","content":str(question)}]})
                except httpx.ReadTimeout:
                    if retry == retries:
                        raise
                else:
                    if fail_zero and question == 0:
                        raise ValueError("persistent evaluation failure on question zero")
                    return response.json()["score"]

    selector = ModelSelector(method="ucb", objective_mode="single", seed=42, tracker=tracker,
        agent=Agent, models={"solver":["metered"]}, eval_fn=lambda _, answer: answer,
        dataset=[(i, None) for i in range(count)], observation_budget_fraction=fraction,
        model_prices={"metered":{"input_price":1, "output_price":2}}, max_search_cost_usd=limit)
    updates = []
    try:
        result = selector.select_best(max_concurrent=slots, on_update=updates.append)
    finally:
        asyncio.run(client.aclose())
    (tmp_path/"requests.json").write_text(json.dumps(requests, indent=2))
    (tmp_path/"result.json").write_text(result.model_dump_json(indent=2))
    (tmp_path/"ledger.json").write_text(json.dumps([{
        "cell": r.data_id, "status": r.status_code, "charge_unknown": r.transport_charge_unknown,
        "not_sent": r.request_not_sent, "input": r.prompt_tokens, "output": r.completion_tokens
    } for r in tracker.get_records()], indent=2))
    payload = json.loads((tmp_path/"result.json").read_text())
    assert updates[-1].model_dump(mode="json") == payload
    return requests, result, payload


@pytest.mark.parametrize("partial_failure", [False, True])
def test_original_question_score_and_usage(tmp_path, partial_failure):
    requests, result, payload = exercise(tmp_path, count=4, slots=4 if partial_failure else 1,
        fraction=1 if partial_failure else .25, fail_zero=partial_failure)
    expected_questions = [1, 2, 3] if partial_failure else [3]
    assert sorted({r["question"] for r in requests}) == ([0, 1, 2, 3] if partial_failure else [3])
    calls = {r["question"]: r for r in requests}
    expected = [(q, calls[q]["score"], {"metered":calls[q]["input_tokens"]},
                 {"metered":calls[q]["output_tokens"]}) for q in expected_questions]
    assert [(d.datapoint_index, d.score, d.input_tokens, d.output_tokens)
            for d in result.results[0].datapoint_results] == expected
    assert [(d["datapoint_index"], d["score"], d["input_tokens"], d["output_tokens"])
            for d in payload["results"][0]["datapoint_results"]] == expected
    if partial_failure:
        assert result.stop_reason == "evaluation_failed"


@pytest.mark.parametrize("case,count,slots,timeouts,retries,limit,expected_cells,known,unknown", [
    ("unrecovered", 1, 1, 1, 0, None, 0, 0, 1),
    ("retry", 1, 1, 1, 1, None, 1, .0001, 1),
    ("multiple_timeouts", 1, 1, 2, 2, None, 1, .0001, 1),
    ("capped", 4, 1, 1, 1, .00025, 1, .0001, 1),
    ("concurrent", 8, 4, 1, 1, .01, 4, .0004, 1),
    ("separate_attempts", 4, 4, 2, 0, None, 3, .0003, 2),
    ("pre_send", 1, 1, 0, 0, .00025, 0, 0, 0),
])
def test_timeout_accounting(tmp_path, case, count, slots, timeouts, retries, limit, expected_cells, known, unknown):
    requests, result, payload = exercise(tmp_path, count=count, slots=slots, timeouts=timeouts,
        retries=retries, limit=limit, pre_send=case=="pre_send", synchronize=case=="concurrent")
    if case == "pre_send":
        assert requests == []
        ledger = json.loads((tmp_path / "ledger.json").read_text())
        assert len(ledger) == 1 and ledger[0]["not_sent"] and not ledger[0]["charge_unknown"]
    elif case == "concurrent":
        assert Counter(r["question"] for r in requests) == {3:2, 4:1, 2:1, 7:1}
    elif case == "separate_attempts":
        assert Counter(r["question"] for r in requests) == {3:2, 0:1, 1:1, 2:1}
    else:
        assert [r["question"] for r in requests] == [3 if count == 4 else 0] * (timeouts + expected_cells)
    assert result.total_evaluations == payload["total_evaluations"] == expected_cells
    assert sorted(d.datapoint_index for model in result for d in model.datapoint_results) == sorted(
        r["question"] for r in requests if r["outcome"] == "response")
    assert result.selection_cost == payload["selection_cost"] == pytest.approx(known)
    assert result.unmeasured_failed_attempts == payload["unmeasured_failed_attempts"] == unknown
    if limit is not None and unknown:
        assert result.stop_reason == "search_cost_unknown"
    elif not expected_cells or case == "separate_attempts":
        assert result.stop_reason == "evaluation_failed"
