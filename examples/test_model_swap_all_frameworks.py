"""
Model-swap smoke test across all example frameworks.

Frameworks covered:
- custom_agent (raw OpenAI SDK)
- openai_agents (OpenAI Agents SDK)
- langchain
- langgraph
- crewai
- ag2

Dependencies are optional per framework; missing packages are skipped.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple

from openai import OpenAI

from agentopt import BruteForceModelSelector


DATASET: List[Tuple[Dict[str, str], str]] = [
    ({"question": "What is 3 + 5? Return only the number."}, "8"),
]


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def _eval_fn(expected: str, actual: Any) -> float:
    text = _to_text(actual)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return 1.0 if digits == expected else 0.0
    return 1.0 if expected.lower() in text.lower() else 0.0


def _run_selector(
    name: str,
    agent_fn: Callable[[Dict[str, Any]], Any],
    models: Dict[str, List[Any]],
) -> None:
    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models=models,
        eval_fn=_eval_fn,
        dataset=DATASET,
    )
    results = selector.select_best()
    best_result = results.get_best()
    best = results.get_best_combo()
    if best is None:
        raise RuntimeError(f"{name}: no best combo found")
    if best_result is None or best_result.accuracy <= 0:
        raise RuntimeError(
            f"{name}: best accuracy is {0.0 if best_result is None else best_result.accuracy:.2f}, expected > 0"
        )
    print(f"[{name}] best={best}")


def _test_custom_agent() -> None:
    client = OpenAI()

    def _resolve(candidate: Any) -> str:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            return str(candidate.get("model"))
        return str(candidate)

    def agent_maker(models: Dict[str, Any]):
        model_name = _resolve(models["agent"])
        print(f"[custom_agent selected] {model_name}")

        def run(input_data: Dict[str, str]) -> str:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer with only the final number.",
                    },
                    {"role": "user", "content": input_data["question"]},
                ],
                temperature=0,
            )
            out = _to_text(resp.choices[0].message.content)
            print(f"[custom_agent output] {model_name}: {out!r}")
            return out

        return run

    _run_selector("custom_agent", agent_maker, {"agent": ["gpt-4o-mini", "gpt-4.1"]})


def _test_openai_agents() -> None:
    from agents import Agent, Runner

    def _resolve(candidate: Any) -> str:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            return str(candidate.get("model"))
        return str(candidate)

    def agent_maker(models: Dict[str, Any]):
        model_name = _resolve(models["agent"])
        print(f"[openai_agents selected] {model_name}")
        agent = Agent(
            name="Math",
            model=model_name,
            instructions="Answer with only the final number.",
        )

        def run(input_data: Dict[str, str]) -> str:
            out = Runner.run_sync(agent, input_data["question"]).final_output
            print(f"[openai_agents output] {model_name}: {out!r}")
            return str(out)

        return run

    _run_selector(
        "openai_agents",
        agent_maker,
        {"agent": [{"model": "gpt-4o-mini"}, {"model": "gpt-4.1"}]},
    )


def _test_langchain() -> None:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template(
        "Answer with only the final number.\nQuestion: {question}"
    )

    def agent_maker(models: Dict[str, Any]):
        llm = models["agent"]
        model_name = getattr(llm, "model_name", "unknown")
        print(f"[langchain selected] {model_name}")
        chain = prompt | llm

        def run(input_data: Dict[str, str]) -> str:
            out = _to_text(chain.invoke({"question": input_data["question"]}))
            print(f"[langchain output] {model_name}: {out!r}")
            return out

        return run

    _run_selector(
        "langchain",
        agent_maker,
        {
            "agent": [
                ChatOpenAI(model="gpt-4o-mini", temperature=0),
                ChatOpenAI(model="gpt-4.1", temperature=0),
            ]
        },
    )


def _test_langgraph() -> None:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph

    def agent_maker(models: Dict[str, Any]):
        llm = models["agent"]
        model_name = getattr(llm, "model_name", "unknown")
        print(f"[langgraph selected] {model_name}")

        def solve_node(state: Dict[str, str]) -> Dict[str, str]:
            resp = llm.invoke(
                "Answer with only the final number. "
                f"Question: {state['question']}"
            )
            return {"answer": _to_text(resp)}

        graph = StateGraph(dict)
        graph.add_node("solve", solve_node)
        graph.set_entry_point("solve")
        graph.add_edge("solve", END)
        app = graph.compile()

        def run(input_data: Dict[str, str]) -> str:
            out = app.invoke({"question": input_data["question"]})["answer"]
            print(f"[langgraph output] {model_name}: {out!r}")
            return out

        return run

    _run_selector(
        "langgraph",
        agent_maker,
        {
            "agent": [
                ChatOpenAI(model="gpt-4o-mini", temperature=0),
                ChatOpenAI(model="gpt-4.1", temperature=0),
            ]
        },
    )


def _test_crewai() -> None:
    from crewai import Agent, Crew, LLM, Task

    def agent_maker(models: Dict[str, Any]):
        llm = models["agent"]
        model_name = getattr(llm, "model", str(llm))
        print(f"[crewai selected] {model_name}")
        agent = Agent(
            role="Math Assistant",
            goal="Answer the math question with only a number.",
            backstory="You are concise and accurate.",
            llm=llm,
        )

        def run(input_data: Dict[str, str]) -> str:
            task = Task(
                description=input_data["question"],
                expected_output="single number",
                agent=agent,
            )
            result = Crew(agents=[agent], tasks=[task], tracing=False, verbose=False).kickoff()
            out = str(result).strip()
            print(f"[crewai output] {model_name}: {out!r}")
            return out

        return run

    _run_selector(
        "crewai",
        agent_maker,
        {"agent": [LLM(model="gpt-4o-mini"), LLM(model="gpt-4.1")]},
    )


def _test_ag2() -> None:
    import autogen

    def agent_maker(models: Dict[str, Any]):
        llm_config = models["agent"]
        model_name = llm_config["model"] if isinstance(llm_config, dict) else str(llm_config)
        print(f"[ag2 selected] {model_name}")

        assistant = autogen.AssistantAgent(
            name="assistant",
            system_message=(
                "You are a math assistant. "
                "Answer with only the final number and append TERMINATE."
            ),
            llm_config=llm_config if isinstance(llm_config, dict) else {"model": llm_config},
        )
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=4,
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
            code_execution_config=False,
        )

        def run(input_data: Dict[str, str]) -> str:
            question = input_data["question"].strip()
            chat_result = user_proxy.initiate_chat(
                assistant,
                message=question,
                max_turns=4,
            )
            for msg in reversed(chat_result.chat_history):
                content = msg.get("content")
                if not content:
                    continue
                out = str(content).replace("TERMINATE", "").strip()
                if not out:
                    continue
                if out == question:
                    # AG2 may include echoed user message in history.
                    continue
                print(f"[ag2 output] {model_name}: {out!r}")
                return out
            return ""

        return run

    _run_selector(
        "ag2",
        agent_maker,
        {
            "agent": [
                {"model": "gpt-4o-mini"},
                {"model": "gpt-4.1"},
            ]
        },
    )


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Load .env first.")

    probe = OpenAI(api_key=api_key).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply exactly with API_OK"}],
        max_tokens=5,
        temperature=0,
    )
    print(
        f"[openai probe] completion_id={probe.id} "
        f"output={_to_text(probe.choices[0].message.content)!r}"
    )

    tests = [
        ("custom_agent", _test_custom_agent),
        ("openai_agents", _test_openai_agents),
        ("langchain", _test_langchain),
        ("langgraph", _test_langgraph),
        ("crewai", _test_crewai),
        ("ag2", _test_ag2),
    ]

    ran = 0
    skipped = 0
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            ran += 1
        except ImportError as e:
            print(f"[{name}] skipped (missing dependency: {e})")
            skipped += 1
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            failed += 1

    print(f"\nsummary: ran={ran}, skipped={skipped}, failed={failed}")
    if ran == 0 or failed > 0:
        raise RuntimeError("one or more framework checks failed")
    print("PASS: all available framework checks succeeded.")


if __name__ == "__main__":
    main()
