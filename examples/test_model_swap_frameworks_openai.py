"""
OpenAI model-swap smoke tests across supported frameworks.

Frameworks covered:
- LangChain
- LangGraph
- CrewAI
- OpenAI Agents SDK

Each test uses two model candidates and one datapoint to keep cost low.
Tests skip automatically if a framework dependency is not installed.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Callable, Dict, List, Tuple

import httpx
from openai import OpenAI

from agentopt import BruteForceModelSelector


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content).strip()


def _eval_fn(expected: str, actual: Any) -> float:
    text = _response_to_text(actual)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return 1.0 if digits == expected else 0.0
    return 1.0 if expected.lower() in text.lower() else 0.0


def _make_dataset() -> List[Tuple[Dict[str, str], str]]:
    # Unique nonce per run helps rule out accidental cache artifacts.
    nonce = uuid.uuid4().hex[:10]
    return [
        (
            {
                "question": (
                    f"Compute 3 + 5. Ignore nonce={nonce}. "
                    "Return only the number."
                )
            },
            "8",
        ),
        (
            {
                "question": (
                    f"What is 10 - 3? Ignore nonce={nonce}. "
                    "Return only the number."
                )
            },
            "7",
        ),
        (
            {
                "question": (
                    f"What is 6 / 2? Ignore nonce={nonce}. "
                    "Return only the number."
                )
            },
            "3",
        ),
    ]


def _run_selector(
    name: str, agent_fn: Callable[[Dict[str, Any]], Any], models: Dict[str, List[Any]],
) -> None:
    dataset = _make_dataset()
    selector = BruteForceModelSelector(
        agent_fn=agent_fn,
        models=models,
        eval_fn=_eval_fn,
        dataset=dataset,
    )
    results = selector.select_best()
    best = results.get_best_combo()
    if best is None:
        raise RuntimeError(f"{name}: no best combo produced")
    print(f"[{name}] best={best}")


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
            out = _response_to_text(chain.invoke({"question": input_data["question"]}))
            print(f"[langchain output] {model_name}: {out!r}")
            return out

        return run

    models = {
        "agent": [
            ChatOpenAI(model="gpt-4o-mini", temperature=0),
            ChatOpenAI(model="gpt-4.1", temperature=0),
        ]
    }
    _run_selector("langchain", agent_maker, models)


def _test_langgraph() -> None:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph

    def agent_maker(models: Dict[str, Any]):
        llm = models["agent"]
        model_name = getattr(llm, "model_name", "unknown")
        print(f"[langgraph selected] {model_name}")

        def solve_node(state: Dict[str, str]) -> Dict[str, str]:
            resp = llm.invoke(
                "Answer with only the number. "
                f"Question: {state['question']}"
            )
            return {"answer": _response_to_text(resp)}

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

    models = {
        "agent": [
            ChatOpenAI(model="gpt-4o-mini", temperature=0),
            ChatOpenAI(model="gpt-4.1", temperature=0),
        ]
    }
    _run_selector("langgraph", agent_maker, models)


def _test_crewai() -> None:
    from crewai import Agent, Crew, LLM, Task

    def agent_maker(models: Dict[str, Any]):
        llm = models["agent"]
        model_name = getattr(llm, "model", str(llm))
        print(f"[crewai selected] {model_name}")
        agent = Agent(
            role="Math Assistant",
            goal="Answer math questions with only the final number.",
            backstory="You are concise and precise.",
            llm=llm,
        )

        def run(input_data: Dict[str, str]) -> str:
            task = Task(
                description=input_data["question"],
                expected_output="Single-number final answer",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            out = str(crew.kickoff()).strip()
            print(f"[crewai output] {model_name}: {out!r}")
            return out

        return run

    models = {
        "agent": [
            LLM(model="gpt-4o-mini"),
            LLM(model="gpt-4.1"),
        ]
    }
    _run_selector("crewai", agent_maker, models)


def _test_openai_agents_sdk() -> None:
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

    models = {
        "agent": [{"model": "gpt-4o-mini"}, {"model": "gpt-4.1"}],
    }
    _run_selector("openai_agents", agent_maker, models)


def main() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Load .env first: "
            "`set -a; source ../.env; set +a`."
        )

    probe = OpenAI(api_key=key).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply exactly with API_OK"}],
        max_tokens=5,
        temperature=0,
    )
    print(
        f"[openai probe] completion_id={probe.id} "
        f"output={(probe.choices[0].message.content or '').strip()!r}"
    )

    # HTTP hook: prints request IDs for real OpenAI HTTP calls made by frameworks.
    original_sync_send = httpx.Client.send
    original_async_send = httpx.AsyncClient.send

    def _sync_send_with_log(self, request, *args, **kwargs):
        response = original_sync_send(self, request, *args, **kwargs)
        if "api.openai.com" in str(request.url):
            req_id = response.headers.get("x-request-id") or response.headers.get(
                "request-id"
            )
            print(
                f"[openai http] {request.method} {request.url.path} "
                f"status={response.status_code} request_id={req_id}"
            )
        return response

    async def _async_send_with_log(self, request, *args, **kwargs):
        response = await original_async_send(self, request, *args, **kwargs)
        if "api.openai.com" in str(request.url):
            req_id = response.headers.get("x-request-id") or response.headers.get(
                "request-id"
            )
            print(
                f"[openai http] {request.method} {request.url.path} "
                f"status={response.status_code} request_id={req_id}"
            )
        return response

    httpx.Client.send = _sync_send_with_log
    httpx.AsyncClient.send = _async_send_with_log

    tests = [
        ("langchain", _test_langchain),
        ("langgraph", _test_langgraph),
        ("crewai", _test_crewai),
        ("openai_agents", _test_openai_agents_sdk),
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

    print("\n=== SUMMARY ===")
    print(f"ran={ran} skipped={skipped} failed={failed}")
    if ran == 0:
        raise RuntimeError("No framework tests ran.")
    if failed > 0:
        raise RuntimeError("One or more framework tests failed.")
    print("PASS: framework model-swap tests completed.")


if __name__ == "__main__":
    main()
