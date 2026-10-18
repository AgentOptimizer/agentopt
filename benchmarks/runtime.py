"""Answer validation and bounded tool execution for benchmark agents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import re
from typing import Any


def response_stop_reason(response: Any) -> str | None:
    metadata = getattr(response, "response_metadata", {}) or {}
    extra = getattr(response, "additional_kwargs", {}) or {}
    for source in (metadata, extra):
        for key in ("stopReason", "stop_reason", "finish_reason"):
            if source.get(key):
                return str(source[key])
    return None


def answer_diagnostics(text: str, stop_reason: str | None) -> dict[str, Any]:
    return {
        "stop_reason": stop_reason,
        "empty_answer": not text.strip(),
        "output_truncated": stop_reason in ("max_tokens", "length", "max_output_tokens"),
    }


def _unwrap_markdown(text: str) -> str:
    """Remove paired bold/code wrappers while preserving unmatched markers."""
    text = text.strip()
    while True:
        for wrapper in ("**", "`"):
            if (len(text) >= 2 * len(wrapper)
                    and text.startswith(wrapper) and text.endswith(wrapper)
                    and wrapper not in text[len(wrapper):-len(wrapper)]):
                text = text[len(wrapper):-len(wrapper)].strip()
                break
        else:
            return text


def parse_math_answer(text: str, stop_reason: str | None = None) -> dict[str, Any]:
    """Read explicit ANSWER lines only; never infer a letter from prose/history."""
    diagnostics = answer_diagnostics(text, stop_reason)
    # Wrappers can surround the whole line, the label (with or without its
    # colon), or the option. Normalize only those boundaries, never prose.
    markers = []
    for line in text.splitlines():
        if not line.strip():
            continue
        line = _unwrap_markdown(line)
        line = re.sub(
            r"^(?:\*\*\s*(ANSWER\s*:?)\s*\*\*|`\s*(ANSWER\s*:?)\s*`)",
            lambda match: match.group(1) or match.group(2), line, flags=re.I,
        )
        markers.append(re.fullmatch(r"ANSWER\s*:\s*(.*)", line, re.I))
    payloads = [_unwrap_markdown(match.group(1)).lower() for match in markers if match]
    options = {value for value in payloads if re.fullmatch(r"[a-e]", value)}
    if diagnostics["output_truncated"]:
        failure = "output_truncated"
    elif stop_reason in ("tool_use", "tool_calls", "function_call"):
        failure = "unfinished_tool_call"
    elif diagnostics["empty_answer"]:
        failure = "empty_answer"
    elif not payloads:
        failure = "missing_answer"
    elif len(options) > 1:
        failure = "conflicting_answers"
    elif any(not re.fullmatch(r"[a-e]", value) for value in payloads):
        failure = "invalid_option"
    elif not markers[-1]:
        failure = "answer_not_final_line"
    else:
        failure = None
    return {
        "answer_text": text.strip(),
        "answer_option": next(iter(options)) if failure is None else None,
        "format_valid": failure is None,
        "format_failure": failure,
        **diagnostics,
    }


@dataclass(frozen=True)
class ToolLimits:
    max_tool_rounds: int | None = 8
    max_tool_calls_per_answer: int = 12
    max_tool_calls_per_question: int = 24
    max_identical_tool_calls: int = 3
    max_consecutive_tool_errors: int = 3

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "max_tool_rounds" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class ToolRunState:
    """One instance per question, shared across its answer/reflection rounds."""

    calls: int = 0
    consecutive_errors: int = 0
    signatures: Counter = field(default_factory=Counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {"tool_call_count": self.calls, "tool_events": list(self.events)}


def add_tool_limit_args(parser):
    descriptions = {
        "max_tool_rounds": "Tool-use batches per answer generation",
        "max_tool_calls_per_answer": "Tool calls per answer generation",
        "max_tool_calls_per_question": "Tool calls across all revisions of one question",
        "max_identical_tool_calls": "Executions of the same tool and arguments per question",
        "max_consecutive_tool_errors": "Consecutive tool errors before requesting a final answer",
    }
    for name, description in descriptions.items():
        default = getattr(ToolLimits(), name)
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default,
                            help=f"{description} (default: {default})")


def tool_limits_from_args(args):
    return ToolLimits(**{name: getattr(args, name) for name in ToolLimits.__dataclass_fields__})


def run_with_tools(
    llm, messages, tools, *, use_tools=True, limits=None, state=None,
    prepare_messages=lambda messages: messages,
):
    """Return history and the last model response, completing every tool call ID.

    A blocked batch is followed by one invocation without tools. Budgets persist
    across revisions through ``state``; per-answer limits reset on each entry.
    """
    from langchain_core.messages import HumanMessage, ToolMessage

    limits = limits or ToolLimits()
    state = state if state is not None else ToolRunState()
    messages = list(messages)
    model = llm.bind_tools(tools) if use_tools else llm
    response = model.invoke(prepare_messages(messages))
    messages.append(response)
    by_name = {tool.name: tool for tool in tools}
    rounds = 0
    answer_calls = 0
    while getattr(response, "tool_calls", None):
        stop = None
        for call in response.tool_calls:
            name, args = call["name"], call["args"]
            signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            reason = stop
            if reason is None and state.calls >= limits.max_tool_calls_per_question:
                reason = "question_tool_limit"
            if reason is None and answer_calls >= limits.max_tool_calls_per_answer:
                reason = "answer_tool_limit"
            if reason is None and state.signatures[signature] >= limits.max_identical_tool_calls:
                reason = "repeated_tool_call"
            if reason is None and state.consecutive_errors >= limits.max_consecutive_tool_errors:
                reason = "consecutive_tool_errors"
            if reason:
                stop = reason
                result = f"Error: {reason}. Finish using the available results; do not call tools again."
                state.events.append({"reason": reason, "tool": name, "args": args, "blocked": True})
                failed = True
            else:
                state.calls += 1
                answer_calls += 1
                state.signatures[signature] += 1
                tool = by_name.get(name)
                if tool is None:
                    result = f"Error: unknown tool '{name}'. Use a supported tool name."
                    error_reason = "unknown_tool"
                else:
                    error_reason = "tool_error"
                    try:
                        result = str(tool.invoke(args))
                    except Exception as exc:
                        result = f"Error: {type(exc).__name__}: {exc}"
                failed = result.lstrip().lower().startswith("error:")
                state.consecutive_errors = state.consecutive_errors + 1 if failed else 0
                if failed:
                    state.events.append({"reason": error_reason, "tool": name, "args": args,
                                         "error": result, "blocked": False})
            messages.append(ToolMessage(content=result, tool_call_id=call["id"],
                                        status="error" if failed else "success"))
        rounds += 1
        if stop is None:
            if state.calls >= limits.max_tool_calls_per_question:
                stop = "question_tool_limit"
            elif answer_calls >= limits.max_tool_calls_per_answer:
                stop = "answer_tool_limit"
            elif state.consecutive_errors >= limits.max_consecutive_tool_errors:
                stop = "consecutive_tool_errors"
            elif limits.max_tool_rounds is not None and rounds >= limits.max_tool_rounds:
                stop = "tool_round_limit"
            if stop:
                state.events.append({"reason": stop, "blocked": False})
        if stop:
            messages.append(HumanMessage(content=(
                f"Tool use stopped: {stop}. Do not call tools again. Using the results "
                "already available, return your final answer in the required format now."
            )))
            response = llm.invoke(prepare_messages(messages))
            messages.append(response)
            # Some providers can still emit calls without tools bound. Do not
            # execute them or leave unmatched call IDs in a later critic turn.
            for call in getattr(response, "tool_calls", None) or []:
                messages.append(ToolMessage(content="Error: tools are disabled after the limit.",
                                            tool_call_id=call["id"], status="error"))
                state.events.append({"reason": "tool_call_after_stop", "tool": call["name"],
                                     "args": call["args"], "blocked": True})
            break
        response = model.invoke(prepare_messages(messages))
        messages.append(response)
    return messages, response
