"""Offline regressions for answer ownership, format failures and tool loops."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import benchmarks.MathQA.eval as mathqa
import benchmarks.HotpotQA.eval as hotpot
from benchmarks.runtime import ToolLimits, ToolRunState, parse_math_answer, run_with_tools
from experiments import run_missing_cells as runner


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.histories = []

    def invoke(self, messages):
        self.histories.append(list(messages))
        return next(self.responses)


class ToolLLM:
    def __init__(self, calls, final="ANSWER: c"):
        self.calls = calls
        self.bound_calls = 0
        self.final_calls = 0
        self.final_history = []
        self.final = final

    def bind_tools(self, tools):
        def invoke(messages):
            self.bound_calls += 1
            calls = self.calls(self.bound_calls)
            return AIMessage(content="", tool_calls=calls)
        return SimpleNamespace(invoke=invoke)

    def invoke(self, messages):
        self.final_calls += 1
        self.final_history = list(messages)
        return AIMessage(content=self.final)


@tool
def numeric(value: int) -> str:
    """Return an integer as text."""
    return str(value)


def call(name="numeric", args=None, id="call"):
    return {"name": name, "args": {"value": 1} if args is None else args, "id": id}


@pytest.mark.parametrize("text,option", [
    ("Final Answer:\nANSWER: c", "c"),
    ("Work\n**ANSWER: D**", "d"),
    ("Work\n  answer: B  ", "b"),
    ("`ANSWER: e`", "e"),
    ("ANSWER: a\nANSWER: a", "a"),
])
def test_explicit_option_does_not_match_heading_or_word_prefix(text, option):
    parsed = parse_math_answer(text)
    assert parsed["format_valid"]
    assert parsed["answer_option"] == option
    assert mathqa.eval_fn(option, text)


@pytest.mark.parametrize("text,failure", [
    ("", "empty_answer"),
    ("The answer is c", "missing_answer"),
    ("a", "missing_answer"),
    ("ANSWER: CORRECT", "invalid_option"),
    ("ANSWER: apple", "invalid_option"),
    ("ANSWER: f", "invalid_option"),
    ("ANSWER: a or b", "invalid_option"),
    ("ANSWER: a\nANSWER: c", "conflicting_answers"),
    ("ANSWER: c\nMore explanation", "answer_not_final_line"),
])
def test_format_failures_are_separate_from_wrong_choices(text, failure):
    parsed = parse_math_answer(text)
    assert parsed["format_failure"] == failure
    assert parsed["answer_option"] is None
    assert not mathqa.eval_fn("c", text)


@pytest.mark.parametrize("factory", [mathqa._mathqa_agent_fn_raw, mathqa._mathqa_agent_fn_langgraph])
@pytest.mark.parametrize("last,reason", [("ANSWER: b", None), ("No choice", None), ("", "max_tokens")])
def test_only_last_answerer_response_is_scored(monkeypatch, factory, last, reason):
    answer = SequenceLLM([AIMessage(content="ANSWER: a"),
                          AIMessage(content=last, response_metadata={"stopReason": reason})])
    critic = SequenceLLM([AIMessage(content="INCORRECT: revise"),
                          AIMessage(content="CORRECT\nANSWER: c")])
    monkeypatch.setattr(mathqa, "make_llm", lambda name: answer if name == "answer" else critic)
    monkeypatch.setattr(mathqa, "supports_tool_calling", lambda name: False)
    result = factory({"answer": "answer", "critic": "critic"})(
        {"messages": [{"role": "user", "content": "Question"}]})
    assert result["answer_text"] == last
    assert result["final"] == last
    assert result["answer_iteration"] == result["iteration"] == 2
    assert len(result["answer_rounds"]) == 2
    assert result["answer_rounds"][0]["answer_option"] == "a"
    assert result["answer_option"] == ("b" if last == "ANSWER: b" else None)
    assert not mathqa.eval_fn("a", result)
    assert not mathqa.eval_fn("c", result)
    assert mathqa.eval_fn("b", result) == (last == "ANSWER: b")
    if reason:
        assert result["output_truncated"] and result["empty_answer"]
        assert result["format_failure"] == "output_truncated"


def test_legacy_message_history_cannot_supply_critic_answer():
    assert not mathqa.eval_fn("c", {"messages": [AIMessage(content="ANSWER: c")]})
    assert not mathqa.eval_fn("c", {"answer_text": "ANSWER: c", "stop_reason": "max_tokens"})
    assert not mathqa.eval_fn("a", {"answer_text": "", "final": "ANSWER: a"})


def test_text_beside_an_unfinished_tool_call_is_not_a_final_answer():
    response = AIMessage(content="ANSWER: c", tool_calls=[call()])
    result = mathqa._answer_snapshot(response, 1)
    assert result["format_failure"] == "unfinished_tool_call"
    assert not mathqa.eval_fn("c", result)


def test_repeated_tool_call_stops_and_completes_all_ids():
    llm = ToolLLM(lambda n: [call(id=f"call-{n}")])
    state = ToolRunState()
    messages, response = run_with_tools(llm, [], [numeric], state=state)
    assert state.calls == 3
    assert llm.bound_calls == 4
    assert llm.final_calls == 1
    assert response.content == "ANSWER: c"
    results = [m for m in messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == [f"call-{n}" for n in range(1, 5)]
    assert results[-1].status == "error"
    assert state.events[-1]["reason"] == "repeated_tool_call"
    assert isinstance(llm.final_history[-1], HumanMessage)


def test_signature_normalization_ignores_argument_key_order():
    llm = ToolLLM(lambda n: [call(args={"a": 1, "b": 2} if n % 2 else {"b": 2, "a": 1})])
    state = ToolRunState()
    # Unknown arguments fail, but the repeat cap can be isolated from the error cap.
    run_with_tools(llm, [], [numeric], state=state,
                   limits=ToolLimits(max_identical_tool_calls=2, max_consecutive_tool_errors=10))
    assert state.calls == 2
    assert state.events[-1]["reason"] == "repeated_tool_call"


def test_parallel_tool_batch_cannot_exceed_answer_limit():
    llm = ToolLLM(lambda n: [call(args={"value": i}, id=str(i)) for i in range(5)])
    state = ToolRunState()
    messages, _ = run_with_tools(llm, [], [numeric], state=state,
                                 limits=ToolLimits(max_tool_calls_per_answer=2))
    assert state.calls == 2
    results = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(results) == 5
    assert [m.status for m in results] == ["success", "success", "error", "error", "error"]


def test_question_budget_persists_across_answer_revisions():
    limits = ToolLimits(max_tool_calls_per_answer=2, max_tool_calls_per_question=3,
                        max_identical_tool_calls=10)
    state = ToolRunState()
    for _ in range(2):
        llm = ToolLLM(lambda n: [call(args={"value": n}, id=str(n))])
        run_with_tools(llm, [], [numeric], state=state, limits=limits)
    assert state.calls == 3
    assert state.events[-1]["reason"] == "question_tool_limit"


@pytest.mark.parametrize("factory", [mathqa._mathqa_agent_fn_raw, mathqa._mathqa_agent_fn_langgraph])
def test_agent_shares_budget_across_revisions_but_resets_for_next_question(monkeypatch, factory):
    answer = ToolLLM(lambda n: [call(id=str(n))])
    critic = SequenceLLM([AIMessage(content=text) for text in
                          ("INCORRECT: revise", "CORRECT", "INCORRECT: revise", "CORRECT")])
    monkeypatch.setattr(mathqa, "make_llm", lambda name: answer if name == "answer" else critic)
    monkeypatch.setattr(mathqa, "supports_tool_calling", lambda name: True)
    monkeypatch.setattr(mathqa, "CALC_TOOLS", [numeric])
    run = factory({"answer": "answer", "critic": "critic"},
                  tool_limits=ToolLimits(max_tool_calls_per_question=1))
    for _ in range(2):
        result = run({"messages": [HumanMessage(content="Q")]})
        assert result["iteration"] == 2
        assert result["tool_call_count"] == 1
        assert result["tool_events"][-1]["reason"] == "question_tool_limit"


@pytest.mark.parametrize("tool_call", [call("unknown"), call(args={"value": "not-an-int"})])
def test_unknown_tools_and_invalid_arguments_have_error_results(tool_call):
    llm = ToolLLM(lambda n: [dict(tool_call, id=str(n))])
    state = ToolRunState()
    messages, _ = run_with_tools(llm, [], [numeric], state=state)
    assert state.calls == 3
    assert state.events[-1]["reason"] == "consecutive_tool_errors"
    assert all(m.status == "error" for m in messages if isinstance(m, ToolMessage))
    assert any("Error:" in event.get("error", "") for event in state.events)


def test_tool_error_strings_and_round_limits_finish_with_tools_disabled():
    @tool
    def broken(value: int) -> str:
        """Return a tool failure."""
        return "Error: undefined variable"
    llm = ToolLLM(lambda n: [call("broken", {"value": n}, str(n))])
    state = ToolRunState()
    run_with_tools(llm, [], [broken], state=state, limits=ToolLimits(max_tool_rounds=1))
    assert state.calls == 1
    assert state.events[-1]["reason"] == "tool_round_limit"
    assert llm.final_calls == 1


def test_calls_emitted_after_forced_finish_are_not_executed():
    llm = ToolLLM(lambda n: [call(id=str(n))])
    llm.invoke = lambda messages: AIMessage(content="", tool_calls=[call(id="late")])
    state = ToolRunState()
    messages, _ = run_with_tools(llm, [], [numeric], state=state,
                                 limits=ToolLimits(max_tool_rounds=1))
    assert state.calls == 1
    assert state.events[-1]["reason"] == "tool_call_after_stop"
    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].tool_call_id == "late"


@pytest.mark.parametrize("factory", [hotpot._hotpotqa_agent_fn_raw, hotpot._hotpotqa_agent_fn_langgraph])
@pytest.mark.parametrize("planner_text,source,invoked", [
    ("terminate()\nFINAL: Paris", "planner", False),
    ("NEXT: find the city", "solver", True),
    ("terminate()", "solver", True),
])
def test_hotpot_records_answer_source_without_changing_planner_policy(
    monkeypatch, factory, planner_text, source, invoked,
):
    planner = SequenceLLM([AIMessage(content=planner_text)])
    solver = SequenceLLM([AIMessage(content="Paris")])
    monkeypatch.setattr(hotpot, "make_llm", lambda name: planner if name == "planner" else solver)
    monkeypatch.setattr(hotpot, "supports_tool_calling", lambda name: False)
    result = factory({"planner": "planner", "solver": "solver"})({"input": "Question"})
    assert result["final"] == "Paris"
    assert result["answer_source"] == source
    assert result["solver_invoked"] is invoked
    assert len(solver.histories) == int(invoked)
    assert hotpot._extract_text(result) == "Paris"


@pytest.mark.parametrize("text,reason,empty,long", [
    ("", "max_tokens", True, False), ("x" * 1001, "end_turn", False, True),
])
def test_hotpot_flags_empty_truncated_and_long_solver_outputs(monkeypatch, text, reason, empty, long):
    planner = SequenceLLM([AIMessage(content="terminate()")])
    solver = SequenceLLM([AIMessage(content=text, response_metadata={"stopReason": reason})])
    monkeypatch.setattr(hotpot, "make_llm", lambda name: planner if name == "planner" else solver)
    monkeypatch.setattr(hotpot, "supports_tool_calling", lambda name: False)
    result = hotpot._hotpotqa_agent_fn_raw({"planner": "planner", "solver": "solver"})({"input": "Q"})
    assert result["final"] == text  # never fall back to the planner's instructions
    assert result["empty_answer"] is empty
    assert result["long_answer"] is long
    assert result["output_truncated"] == (reason == "max_tokens")


def test_planner_final_uses_its_own_truncation_metadata(monkeypatch):
    planner = SequenceLLM([AIMessage(content="terminate()\nFINAL: Paris",
                                     response_metadata={"finish_reason": "length"})])
    monkeypatch.setattr(hotpot, "make_llm", lambda name: planner)
    monkeypatch.setattr(hotpot, "supports_tool_calling", lambda name: False)
    result = hotpot._hotpotqa_agent_fn_raw({"planner": "P", "solver": "S"})({"input": "Q"})
    assert result["answer_source"] == "planner"
    assert result["output_truncated"]
    assert not result["solver_invoked"]


def test_hotpot_unknown_tool_is_paired_with_error_instead_of_silently_ignored():
    llm = ToolLLM(lambda n: [call("unknown", id=str(n))], final="Paris")
    diagnostics = {}
    final, messages = hotpot._invoke_with_tools(llm, [], diagnostics=diagnostics)
    assert final == "Paris"
    assert len([m for m in messages if isinstance(m, ToolMessage)]) == 3


def test_supplement_persists_diagnostics_in_new_output_only(monkeypatch, tmp_path):
    import agentopt

    class Tracker:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def flush_cache(self): pass
        def track(self, **kwargs):
            from contextlib import nullcontext
            return nullcontext()

    actual = {**parse_math_answer("ANSWER: z"), "final": "ANSWER: z",
              "answer_source": "answerer", "answer_iteration": 1,
              "tool_events": [{"reason": "repeated_tool_call"}], "tool_call_count": 3}
    monkeypatch.setattr(agentopt, "LLMTracker", Tracker)
    monkeypatch.setattr(runner, "load_dataset", lambda name: [({}, "c")] * 200)
    monkeypatch.setattr(runner, "build_agent", lambda *a, **k: lambda data: actual)
    monkeypatch.setattr(runner, "tracked_result_metrics", lambda *a, **k: {"cost": 0.01})
    runner.run_cells([{"benchmark": "mathqa", "configuration_id": "test", "question_id": 0,
                       "role1_model": "A", "role2_model": "B"}], output_dir=tmp_path,
                      max_concurrent=1, aws_context={"account": "test", "region": "test",
                                                   "project": "test", "billing-tag1": "test"})
    row = json.loads((tmp_path / "mathqa.jsonl").read_text(encoding="utf-8"))
    assert row["score"] == 0
    assert row["status"] == "ok"
    assert row["terminal_reason"] == "invalid_option"
    assert row["answer_option"] is None
    assert row["tool_events"] == actual["tool_events"]
    assert row["evaluation_protocol"] == "guarded_answers_v1"
    assert row["tool_limits"]["max_tool_calls_per_question"] == 24
    assert runner.DEFAULT_OUTPUT_DIR.name == "output_guarded"


@pytest.mark.parametrize("field", ToolLimits.__dataclass_fields__)
def test_all_tool_limits_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        replace(ToolLimits(), **{field: 0})
