from agentopt.model_selection.streaming_brute_force import (
    StreamingBruteForceModelSelector,
)


def _agent_fn(combo):
    model_name = combo["agent"]

    def _run(input_data):
        x = input_data["x"]
        return x if model_name == "good" else x + 1

    return _run


def _eval_fn(expected, actual):
    return 1.0 if expected == actual else 0.0


def test_streaming_update_accumulates_and_tracks_best():
    selector = StreamingBruteForceModelSelector(
        agent_fn=_agent_fn,
        models={"agent": ["good", "bad"]},
        eval_fn=_eval_fn,
        dataset=[({"x": 0}, 0)],
    )

    selector.update([({"x": 1}, 1)])
    selector.update([({"x": 2}, 2)])
    results = selector.results()

    by_name = {r.model_name: r for r in results}
    assert by_name["agent=good"].accuracy == 1.0
    assert by_name["agent=bad"].accuracy == 0.0
    assert by_name["agent=good"].is_best
    assert len(by_name["agent=good"].datapoint_results) == 2
    assert len(by_name["agent=bad"].datapoint_results) == 2


def test_select_best_consumes_seed_dataset_once():
    selector = StreamingBruteForceModelSelector(
        agent_fn=_agent_fn,
        models={"agent": ["good", "bad"]},
        eval_fn=_eval_fn,
        dataset=[({"x": 3}, 3)],
    )

    first = selector.select_best()
    second = selector.select_best()

    first_by_name = {r.model_name: r for r in first}
    second_by_name = {r.model_name: r for r in second}

    assert len(first_by_name["agent=good"].datapoint_results) == 1
    assert len(second_by_name["agent=good"].datapoint_results) == 1
