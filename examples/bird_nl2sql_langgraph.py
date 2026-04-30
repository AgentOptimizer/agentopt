"""
Minimal LangGraph NL2SQL example for the BIRD benchmark.

Prerequisites:
    1. uv sync --extra dev --extra langgraph
    2. uv run python benchmarks/bird/setup_bird.py
    3. Set OPENAI_API_KEY before running without --dry-run

The default invocation runs one BIRD question and calls an LLM. It uses the
same model for initial generation and refinement; AgentOpt is responsible for
choosing models around this agent. Use --dry-run to inspect the prompt and
resolved database path without making an LLM call.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TypedDict

from agentopt.benchmarks.bird import (
    DEFAULT_BIRD_DATA_DIR,
    BirdExample,
    bird_db_path,
    build_bird_prompt,
    evaluate_sql,
    execute_sql,
    extract_sql,
    load_bird_examples,
    read_sqlite_schema,
)


class NL2SQLState(TypedDict, total=False):
    messages: list[Any]
    prompt: str
    db_path: str
    gold_sql: str
    raw_response: str
    sql: str
    rows: list[Any] | None
    execution_error: str | None
    evaluation: dict[str, Any] | None
    attempts: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    feedback: str
    refinement_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_BIRD_DATA_DIR,
        help=f"BIRD dev data directory. Default: {DEFAULT_BIRD_DATA_DIR}",
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=0,
        help="BIRD question_id to run. Default: 0",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="ChatOpenAI model name. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "Optional environment variable name for the API key. Defaults to "
            "OPENROUTER_API_KEY for OpenRouter base URLs, otherwise ChatOpenAI "
            "uses its normal OPENAI_API_KEY lookup."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature. Default: 0.0",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="SQLite query timeout. Default: 30",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="LLM request timeout. Default: 120",
    )
    parser.add_argument(
        "--max-schema-chars",
        type=int,
        default=20000,
        help="Truncate schema text above this many characters. Default: 20000",
    )
    parser.add_argument(
        "--max-rows-print",
        type=int,
        default=10,
        help="Rows to include in stdout/output JSON. Default: 10",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip gold SQL execution/result comparison.",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=1,
        help="Maximum same-model refinement attempts after the initial SQL. Default: 1",
    )
    parser.add_argument(
        "--no-refine-on-mismatch",
        action="store_true",
        help=(
            "Only refine on SQL execution errors. By default, benchmark result "
            "mismatches can also trigger generic repair feedback."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt and exit without calling an LLM.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def load_example(data_dir: Path, question_id: int) -> BirdExample:
    examples = load_bird_examples(data_dir, question_ids=[question_id], limit=1)
    if not examples:
        raise ValueError(f"No BIRD example found for question_id={question_id}")
    return examples[0]


def import_langgraph_deps():
    try:
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise SystemExit(
            "Missing LangGraph example dependencies. Run: "
            "uv sync --extra dev --extra langgraph"
        ) from exc
    return ChatOpenAI, END, StateGraph


def llm_call_record(node_name: str, model: str, response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}

    input_tokens = usage.get("input_tokens") or token_usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or token_usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens") or token_usage.get("total_tokens")

    return {
        "node": node_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage": usage or token_usage or None,
    }


def build_graph(args: argparse.Namespace):
    ChatOpenAI, END, StateGraph = import_langgraph_deps()
    base_url = args.base_url
    if base_url is None and "/" in args.model:
        base_url = os.environ.get("OPENROUTER_BASE_URL")

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "timeout": args.request_timeout_seconds,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    api_key_env = args.api_key_env
    if api_key_env is None and base_url and "openrouter" in base_url.lower():
        api_key_env = "OPENROUTER_API_KEY"
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"Missing API key environment variable: {api_key_env}")
        llm_kwargs["api_key"] = api_key
    llm = ChatOpenAI(**llm_kwargs)

    def generate_sql_node(state: NL2SQLState) -> dict[str, Any]:
        response = llm.invoke(state["messages"])
        raw_response = str(response.content)
        return {
            "raw_response": raw_response,
            "sql": extract_sql(raw_response),
            "llm_calls": [llm_call_record("generate_sql", args.model, response)],
            "refinement_count": 0,
        }

    def feedback_for_state(state: NL2SQLState) -> str:
        if state.get("execution_error"):
            return (
                "The previous SQL failed when executed by SQLite.\n\n"
                f"SQLite error:\n{state['execution_error']}\n\n"
                "Revise the SQL so it is valid for the schema and question."
            )

        evaluation = state.get("evaluation") or {}
        if evaluation.get("is_correct") is False:
            details = [
                "The previous SQL executed, but its result did not match the "
                "benchmark expected result.",
                f"Reason: {evaluation.get('reason', 'result mismatch')}",
            ]
            if "gold_row_count" in evaluation or "predicted_row_count" in evaluation:
                details.append(
                    "Row counts: expected={expected}, predicted={predicted}".format(
                        expected=evaluation.get("gold_row_count", "?"),
                        predicted=evaluation.get("predicted_row_count", "?"),
                    )
                )
            details.append(
                "Revise the SQL using only the original schema, question, and hints. "
                "Do not assume hidden tables or columns."
            )
            return "\n".join(details)

        return ""

    def refine_sql_node(state: NL2SQLState) -> dict[str, Any]:
        feedback = feedback_for_state(state)
        repair_prompt = (
            "Original NL2SQL prompt:\n"
            f"{state['prompt']}\n\n"
            "Previous SQL:\n"
            f"{state.get('sql', '')}\n\n"
            "Feedback:\n"
            f"{feedback}\n\n"
            "Return only one revised SQLite SQL query."
        )
        response = llm.invoke(
            [
                (
                    "system",
                    "You repair SQLite SQL queries. Return only SQL, no markdown.",
                ),
                ("human", repair_prompt),
            ]
        )
        raw_response = str(response.content)
        llm_calls = list(state.get("llm_calls") or [])
        llm_calls.append(llm_call_record("refine_sql", args.model, response))
        return {
            "raw_response": raw_response,
            "sql": extract_sql(raw_response),
            "feedback": feedback,
            "llm_calls": llm_calls,
            "refinement_count": int(state.get("refinement_count", 0)) + 1,
        }

    def execute_sql_node(state: NL2SQLState) -> dict[str, Any]:
        execution = execute_sql(
            state["db_path"],
            state.get("sql", ""),
            timeout_seconds=args.timeout_seconds,
        )
        evaluation = None
        if not args.no_evaluate and state.get("gold_sql"):
            evaluation = evaluate_sql(
                state["db_path"],
                state.get("sql", ""),
                state["gold_sql"],
                timeout_seconds=args.timeout_seconds,
            )
        attempts = list(state.get("attempts") or [])
        attempts.append(
            {
                "attempt": len(attempts),
                "refinement_count": int(state.get("refinement_count", 0)),
                "sql": state.get("sql", ""),
                "execution_error": execution.error,
                "evaluation": evaluation,
            }
        )
        return {
            "rows": execution.rows,
            "execution_error": execution.error,
            "evaluation": evaluation,
            "attempts": attempts,
        }

    def route_after_execute(state: NL2SQLState) -> str:
        if int(state.get("refinement_count", 0)) >= max(0, args.max_refinements):
            return END

        if state.get("execution_error"):
            return "refine_sql"

        if args.no_evaluate or args.no_refine_on_mismatch:
            return END

        evaluation = state.get("evaluation") or {}
        if evaluation.get("is_correct") is False:
            return "refine_sql"

        return END

    graph = StateGraph(NL2SQLState)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("refine_sql", refine_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.set_entry_point("generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("refine_sql", "execute_sql")
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "refine_sql": "refine_sql",
            END: END,
        },
    )
    return graph.compile()


def result_payload(
    example: BirdExample,
    db_path: Path,
    prompt: str,
    result: NL2SQLState,
    max_rows_print: int,
) -> dict[str, Any]:
    rows = result.get("rows")
    return {
        "example": example.to_dict(),
        "db_path": str(db_path),
        "prompt": prompt,
        "raw_response": result.get("raw_response"),
        "sql": result.get("sql"),
        "execution_error": result.get("execution_error"),
        "rows": rows[:max_rows_print] if isinstance(rows, list) else rows,
        "row_count": len(rows) if isinstance(rows, list) else None,
        "evaluation": result.get("evaluation"),
        "refinement_count": result.get("refinement_count", 0),
        "llm_calls": result.get("llm_calls", []),
        "attempts": result.get("attempts", []),
    }


def main() -> int:
    args = parse_args()
    example = load_example(args.data_dir, args.question_id)
    db_path = bird_db_path(args.data_dir, example.db_id)
    schema = read_sqlite_schema(db_path, max_chars=args.max_schema_chars)
    prompt = build_bird_prompt(example, schema)

    if args.dry_run:
        print(f"question_id: {example.question_id}")
        print(f"db_id: {example.db_id}")
        print(f"db_path: {db_path}")
        print("\n--- Prompt ---")
        print(prompt)
        return 0

    app = build_graph(args)
    result = app.invoke(
        {
            "messages": [
                (
                    "system",
                    "You translate natural language questions to SQLite SQL.",
                ),
                ("human", prompt),
            ],
            "prompt": prompt,
            "db_path": str(db_path),
            "gold_sql": example.gold_sql,
        }
    )
    payload = result_payload(
        example,
        db_path,
        prompt,
        result,
        max_rows_print=args.max_rows_print,
    )

    print(json.dumps(payload, indent=2, default=str))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
