"""Unit tests for StreamingRandomSearchModelSelector."""

from agentopt import ModelSelector, StreamingRandomSearchModelSelector


class DummyAgent:
    """Simple deterministic agent for selector tests."""

    def __init__(self, models):
        self.models = models

    def run(self, _input_data):
        return f"{self.models['planner']}|{self.models['solver']}"


def exact_match(expected, actual):
    return float(expected == actual)


def test_streaming_random_sampling_size():
    selector = StreamingRandomSearchModelSelector(
        agent=DummyAgent,
        models={
            "planner": ["p1", "p2", "p3"],
            "solver": ["s1", "s2"],
        },
        eval_fn=exact_match,
        dataset=[({"q": "seed"}, "p1|s1")],
        sample_fraction=0.5,
        seed=7,
    )
    # 3 * 2 = 6 combos; sample_fraction=0.5 => ceil(3) = 3 sampled combos.
    assert len(selector._sampled_combos) == 3
    assert len(selector._all_combos) == 6


def test_streaming_random_updates_best_combo_over_time():
    selector = StreamingRandomSearchModelSelector(
        agent=DummyAgent,
        models={"planner": ["p1", "p2"], "solver": ["s1", "s2"]},
        eval_fn=exact_match,
        dataset=[({"q": "seed"}, "p1|s1")],
        sample_fraction=1.0,
        seed=3,
    )

    initial = selector.select_best()
    assert initial.get_best_combo() == {"planner": "p1", "solver": "s1"}

    # Stream in data that favors a different combination.
    selector.update(
        [
            ({"q": "b1"}, "p2|s2"),
            ({"q": "b2"}, "p2|s2"),
            ({"q": "b3"}, "p2|s2"),
        ]
    )
    best_after = selector.results().get_best_combo()
    assert best_after == {"planner": "p2", "solver": "s2"}


def test_streaming_random_converges_after_stable_batches():
    selector = StreamingRandomSearchModelSelector(
        agent=DummyAgent,
        models={"planner": ["p1", "p2"], "solver": ["s1", "s2"]},
        eval_fn=exact_match,
        dataset=[({"q": "seed"}, "p2|s2")],
        sample_fraction=1.0,
        seed=11,
    )

    selector.select_best()
    for _ in range(10):
        selector.update_one({"q": "stream"}, "p2|s2")

    assert selector.has_converged() is True
    assert selector.should_continue() is False


def test_model_selector_factory_supports_streaming_random():
    selector = ModelSelector(
        agent=DummyAgent,
        models={"planner": ["p1"], "solver": ["s1"]},
        eval_fn=exact_match,
        dataset=[({"q": "seed"}, "p1|s1")],
        method="streaming_random",
    )
    assert isinstance(selector, StreamingRandomSearchModelSelector)
