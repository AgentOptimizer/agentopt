from collections import Counter

import json
import pickle
import sys
from types import SimpleNamespace

import pytest

from experiments.run_missing_cells import (
    REPO_ROOT,
    account_from_profile_arns,
    build_agent,
    compute_cost,
    context_overflow_result,
    expand_plan,
    is_context_length_exceeded,
    load_plan,
    preflight_aws,
)


ORIGINAL_EXPERIMENT_PRICES = {
    "Claude 3 Haiku": (0.25, 1.25),
    "Claude Haiku 4.5": (1.00, 5.00),
    "Claude Opus 4.6": (5.00, 25.00),
    "Kimi K2.5": (0.60, 3.00),
    "Ministral 3 8B": (0.15, 0.15),
    "Qwen3 32B": (0.15, 0.60),
    "Qwen3 Next 80B A3B": (0.15, 1.20),
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.07, 0.30),
}


class ValidationException(Exception):
    pass


def test_only_explicit_context_validation_errors_are_terminal_incorrect():
    context_error = ValidationException(
        "This model's maximum context length is 32768 tokens; prompt contains "
        "at least 32769 input tokens"
    )
    assert is_context_length_exceeded(context_error)
    assert not is_context_length_exceeded(ValidationException("invalid parameter"))
    assert not is_context_length_exceeded(TimeoutError("read timed out"))


def test_context_overflow_result_preserves_recorded_usage_and_cost():
    class Record:
        def __init__(self, cached):
            self.cached = cached

    class Tracker:
        def get_usage(self, *, data_id):
            assert data_id == "cell"
            return {"Qwen3 32B": (1000, 2000), "DeepSeek R1": (3000, 4000)}

        def get_records(self, *, data_id):
            assert data_id == "cell"
            return [Record(False), Record(True)]

        def get_cached_latency(self, *, data_id):
            return 7.0

        def get_server_latency(self, *, data_id):
            return 1234.0

    exc = ValidationException("maximum context length exceeded")
    row = context_overflow_result(
        {"benchmark": "mathqa"},
        exc=exc,
        tracker=Tracker(),
        data_id="cell",
        wall_seconds=5.0,
    )
    assert row["status"] == "ok"
    assert row["score"] == 0.0
    assert row["terminal_reason"] == "context_length_exceeded"
    assert row["input_tokens"] == {"Qwen3 32B": 1000, "DeepSeek R1": 3000}
    assert row["output_tokens"] == {"Qwen3 32B": 2000, "DeepSeek R1": 4000}
    assert row["cost"] == pytest.approx(0.027)
    assert row["latency_seconds"] == 12.0
    assert row["server_latency_ms"] == 1234.0
    assert row["model_call_count"] == 2
    assert row["cached_call_count"] == 1


def test_9x9_builder_has_expected_model_and_configuration_order():
    from experiments.build_completed_9x9 import (
        MODELS_9,
        expected_configurations,
    )

    assert MODELS_9[-1] == "DeepSeek R1"
    assert len(MODELS_9) == 9
    for benchmark in ("hotpotqa", "mathqa"):
        configurations = expected_configurations(benchmark)
        assert len(configurations) == 81
        assert len(set(configurations)) == 81
        assert sum("DeepSeek R1" in value for value in configurations) == 17


def test_9x9_qwen_context_adjudications_are_incorrect_with_recorded_usage():
    from experiments.build_completed_9x9 import load_adjudications

    path = (
        REPO_ROOT
        / "experiments"
        / "missing_cells"
        / "adjudicated_failures.9x9.json"
    )
    rows = load_adjudications(path, "mathqa")
    config = "answer=Qwen3 32B + critic=DeepSeek R1"
    assert set(rows) == {(config, question_id) for question_id in (65, 93, 108, 121)}
    assert all(row["score"] == 0.0 for row in rows.values())
    assert sum(sum(row["input_tokens"].values()) for row in rows.values()) == 112671
    assert sum(sum(row["output_tokens"].values()) for row in rows.values()) == 139863
    assert sum(row["cost"] for row in rows.values()) == pytest.approx(0.28880445)


def test_deepseek_retry_plans_cover_current_failure_classes():
    transient = expand_plan(
        load_plan(REPO_ROOT / "experiments/missing_cells/retry_deepseek_9x9_transient.csv"),
        "all",
    )
    context = expand_plan(
        load_plan(
            REPO_ROOT
            / "experiments/missing_cells/retry_deepseek_9x9_qwen_context.csv"
        ),
        "all",
    )
    assert len(transient) == 8
    assert len(context) == 10
    assert all(cell["benchmark"] == "mathqa" for cell in transient + context)
    assert all(cell["role1_model"] == "Qwen3 32B" for cell in context)


def test_timeout_retry_plan_contains_only_the_three_unresolved_rows():
    cells = expand_plan(
        load_plan(
            REPO_ROOT
            / "experiments/missing_cells/retry_deepseek_9x9_timeout3.csv"
        ),
        "all",
    )
    assert [cell["question_id"] for cell in cells] == [31, 69, 87]
    assert all(
        cell["configuration_id"]
        == "answer=Ministral 3 8B + critic=DeepSeek R1"
        for cell in cells
    )


def test_mathqa_iteration_override_is_forwarded(monkeypatch):
    import benchmarks.MathQA.eval as mathqa_eval

    captured = {}

    def fake_factory(models, *, max_iterations, max_tool_rounds):
        captured.update(
            models=models,
            max_iterations=max_iterations,
            max_tool_rounds=max_tool_rounds,
        )
        return "agent"

    monkeypatch.setattr(mathqa_eval, "_mathqa_agent_fn_langgraph", fake_factory)
    agent = build_agent(
        {
            "benchmark": "mathqa",
            "role1_model": "Qwen3 32B",
            "role2_model": "DeepSeek R1",
        },
        max_tool_rounds=1,
        mathqa_max_iterations=1,
    )
    assert agent == "agent"
    assert captured == {
        "models": {"answer": "Qwen3 32B", "critic": "DeepSeek R1"},
        "max_iterations": 1,
        "max_tool_rounds": 1,
    }


def test_runner_adds_repo_root_to_module_search_path():
    assert str(REPO_ROOT) in sys.path


def test_plan_expands_to_exact_audited_counts():
    cells = expand_plan(load_plan(), "all")
    assert len(cells) == 1271
    assert Counter(cell["benchmark"] for cell in cells) == {
        "hotpotqa": 32,
        "mathqa": 1239,
    }
    assert len(
        {
            (cell["benchmark"], cell["configuration_id"], cell["question_id"])
            for cell in cells
        }
    ) == len(cells)


def test_supplement_uses_original_experiment_prices():
    import benchmarks.common as common

    for model, (input_price, output_price) in ORIGINAL_EXPERIMENT_PRICES.items():
        assert common.BEDROCK_PRICES[model] == {
            "input_price": input_price,
            "output_price": output_price,
        }

    arn = common._DISPLAY_NAME_TO_ARN["Ministral 3 8B"]
    assert compute_cost({arn: 791}, {arn: 404}) == pytest.approx(0.00017925)


def test_historical_sample_result_shape_is_compatible():
    experiments_dir = REPO_ROOT / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    source = experiments_dir / "results" / "cache_db_results" / "mathqa_lookup.pkl"
    with source.open("rb") as handle:
        lookup = pickle.load(handle)
    config = "answer=Ministral 3 8B + critic=Ministral 3 8B"
    sample = lookup["table"][config][0]
    assert set(vars(sample)) == {
        "score",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "cost",
    }
    assert "SampleResult" in repr(sample)


def test_account_from_profile_arns_validates_scope():
    arn = (
        "arn:aws:bedrock:us-east-1:920736616554:"
        "application-inference-profile/4tipnum3tltv"
    )
    assert account_from_profile_arns([arn], region="us-east-1") == "920736616554"
    with pytest.raises(RuntimeError, match="application inference profile"):
        account_from_profile_arns([arn.replace("us-east-1", "us-west-2")], region="us-east-1")


def test_bearer_preflight_checks_profile_tags_without_sts(monkeypatch):
    arn = (
        "arn:aws:bedrock:us-east-1:920736616554:"
        "application-inference-profile/4tipnum3tltv"
    )

    import benchmarks.common as common

    monkeypatch.setitem(common._DISPLAY_NAME_TO_ARN, "Ministral 3 8B", arn)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")

    class FakeBedrock:
        def list_tags_for_resource(self, *, resourceARN):
            assert resourceARN == arn
            return {
                "tags": [
                    {"key": "project", "value": "agentopt"},
                    {"key": "billing-tag1", "value": "qx2278"},
                ]
            }

    class FakeSession:
        def __init__(self, *, region_name):
            assert region_name == "us-east-1"

        def client(self, service_name):
            assert service_name == "bedrock"
            return FakeBedrock()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=FakeSession))
    result = preflight_aws(
        [{"role1_model": "Ministral 3 8B", "role2_model": "Ministral 3 8B"}],
        region="us-east-1",
        project_tag="agentopt",
        billing_tag1="qx2278",
    )
    assert result["account"] == "920736616554"
    assert result["profiles"] == [arn]
    assert result["region"] == "us-east-1"
    assert result["project"] == "agentopt"
    assert result["billing-tag1"] == "qx2278"


def test_profile_map_override(monkeypatch, tmp_path):
    profile_map = tmp_path / "profiles.json"
    arn = (
        "arn:aws:bedrock:us-east-1:920736616554:"
        "application-inference-profile/4tipnum3tltv"
    )
    profile_map.write_text(
        json.dumps({"Ministral 3 8B": arn}), encoding="utf-8"
    )
    monkeypatch.setenv("AGENTOPT_BEDROCK_PROFILE_MAP", str(profile_map))

    import importlib
    import benchmarks.common as common

    common = importlib.reload(common)
    assert common._DISPLAY_NAME_TO_ARN["Ministral 3 8B"] == arn
    assert common.display_name(arn) == "Ministral 3 8B"
    assert arn in common.BEDROCK_PRICES


def test_new_ministral_profile_supplies_mistral_provider(monkeypatch, tmp_path):
    profile_map = tmp_path / "profiles.json"
    arn = (
        "arn:aws:bedrock:us-east-1:920736616554:"
        "application-inference-profile/4tipnum3tltv"
    )
    profile_map.write_text(
        json.dumps({"Ministral 3 8B": arn}), encoding="utf-8"
    )
    monkeypatch.setenv("AGENTOPT_BEDROCK_PROFILE_MAP", str(profile_map))

    captured = {}

    class FakeChatBedrockConverse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_aws",
        SimpleNamespace(ChatBedrockConverse=FakeChatBedrockConverse),
    )

    import importlib
    import benchmarks.common as common

    common = importlib.reload(common)
    common.make_llm("Ministral 3 8B")
    assert captured["model"] == arn
    assert captured["provider"] == "mistral"
