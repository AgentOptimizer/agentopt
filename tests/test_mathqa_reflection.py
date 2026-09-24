from langchain_core.messages import AIMessage, HumanMessage

import benchmarks.MathQA.eval as mathqa_eval


class _AnswerLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 2:
            assert isinstance(messages[-1], HumanMessage)
            assert "critic feedback" in messages[-1].content
            return AIMessage(content="Answer: b")
        return AIMessage(content="Answer: a")


class _CriticLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="INCORRECT: choose b instead")
        return AIMessage(content="CORRECT")


def _patch_models(monkeypatch, answer, critic):
    monkeypatch.setattr(
        mathqa_eval,
        "make_llm",
        lambda model: answer if model == "answer-model" else critic,
    )
    monkeypatch.setattr(mathqa_eval, "supports_tool_calling", lambda model: False)


def test_raw_reflection_adds_user_turn_before_retry(monkeypatch):
    answer = _AnswerLLM()
    critic = _CriticLLM()
    _patch_models(monkeypatch, answer, critic)

    run = mathqa_eval._mathqa_agent_fn_raw(
        {"answer": "answer-model", "critic": "critic-model"},
        max_iterations=3,
    )
    result = run({"messages": [{"role": "user", "content": "Question"}]})

    assert answer.calls == 2
    assert critic.calls == 2
    assert result["final"] == "Answer: b"


def test_langgraph_reflection_adds_user_turn_before_retry(monkeypatch):
    answer = _AnswerLLM()
    critic = _CriticLLM()
    _patch_models(monkeypatch, answer, critic)

    run = mathqa_eval._mathqa_agent_fn_langgraph(
        {"answer": "answer-model", "critic": "critic-model"},
        max_iterations=3,
    )
    result = run({"messages": [{"role": "user", "content": "Question"}]})

    assert answer.calls == 2
    assert critic.calls == 2
    assert result["final"] == "Answer: b"
