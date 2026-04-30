"""Token-usage extraction from LLM API response bodies.

Three input shapes are handled:

* JSON object — OpenAI/Anthropic/Gemini single-response format.
* JSON array — Gemini's ``streamGenerateContent`` non-SSE format
  (returned without ``alt=sse``).
* SSE stream — ``data: {...}`` frames from any provider's streaming API.

All extraction returns ``(ParsedUsage | None, parse_failure_reason | None)``.
The proxy uses the failure reason to attach a structured error to the
``CallRecord`` so a successful call with unparseable usage isn't silently
reported as zero tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse


# Sentinel model name for a 200 response whose usage we couldn't extract.
PARSE_FAILED_MODEL = "<parse-failed>"


# OpenAI-compatible chat/completions paths.  Used to surface the
# ``stream_options.include_usage`` quirk specifically.
_OPENAI_COMPAT_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/chat/completions",
)


@dataclass(frozen=True)
class ParsedUsage:
    """Token usage extracted from an LLM response."""

    model: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    response_body: dict  # representative chunk — kept for CallRecord debug


# ---------------------------------------------------------------------------
# Streaming detection
# ---------------------------------------------------------------------------


def is_streaming_request(request_body: dict, path_or_url: str) -> bool:
    """Whether the request expects a streamed response.

    Detects:
    * OpenAI / Anthropic — ``"stream": true`` in the body.
    * Gemini — ``:streamGenerateContent`` endpoint, with or without
      ``?alt=sse`` (Gemini doesn't put a stream flag in the body).
    """
    if request_body.get("stream") is True:
        return True
    return "streamGenerateContent" in path_or_url or "alt=sse" in path_or_url


def is_openai_compatible_url(url: str) -> bool:
    path = urlparse(url).path or ""
    return any(path.endswith(p) for p in _OPENAI_COMPAT_PATHS)


def has_include_usage(request_body: dict) -> bool:
    opts = request_body.get("stream_options")
    return isinstance(opts, dict) and opts.get("include_usage") is True


# ---------------------------------------------------------------------------
# Per-provider usage shape — internal helpers
# ---------------------------------------------------------------------------


def _gemini_completion_tokens(usage: dict) -> int:
    """Sum visible output and reasoning tokens for Gemini thinking models.

    Gemini reports ``candidatesTokenCount`` (visible) and
    ``thoughtsTokenCount`` (reasoning) separately; both bill as output, so
    callers want the sum.  Falls back to ``totalTokenCount - promptTokenCount``
    if individual fields aren't broken out.
    """
    candidates = usage.get("candidatesTokenCount") or 0
    thoughts = usage.get("thoughtsTokenCount") or 0
    if candidates or thoughts:
        return candidates + thoughts
    total = usage.get("totalTokenCount")
    prompt = usage.get("promptTokenCount") or 0
    if isinstance(total, int) and total > prompt:
        return total - prompt
    return 0


def _extract_usage_dict(usage: dict) -> Tuple[int, int]:
    """``(prompt_tokens, completion_tokens)`` from any provider's usage dict."""
    prompt = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("promptTokenCount")
        or 0
    )
    completion = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or _gemini_completion_tokens(usage)
    )
    return prompt, completion


def _parse_chunk(chunk: dict) -> Optional[ParsedUsage]:
    """Pull usage from a single response object (any provider)."""
    model = chunk.get("model") or chunk.get("modelVersion")
    usage = chunk.get("usage") or chunk.get("usageMetadata")
    msg = chunk.get("message")
    if not usage and isinstance(msg, dict):
        usage = msg.get("usage")
        model = model or msg.get("model")
    if not model or not usage:
        return None
    prompt, completion = _extract_usage_dict(usage)
    return ParsedUsage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        response_body=chunk,
    )


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------


def extract_usage(body_bytes: bytes,) -> Tuple[Optional[ParsedUsage], Optional[str]]:
    """Parse usage from a non-streaming response body.

    Returns ``(usage, parse_failure_reason)``: exactly one is non-None.
    """
    if not body_bytes:
        return None, "response body was empty"

    try:
        parsed = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return (
            None,
            (
                f"response body is not valid JSON ({type(exc).__name__}); "
                f"first 200 bytes: {body_bytes[:200]!r}"
            ),
        )

    if isinstance(parsed, dict):
        usage = _parse_chunk(parsed)
        if usage is None:
            return (
                None,
                (
                    "response body is a JSON object but has no recognized "
                    "usage/model fields (expected one of: usage, usageMetadata, "
                    "input_tokens/output_tokens, prompt_tokens/completion_tokens, "
                    "promptTokenCount/candidatesTokenCount). "
                    f"keys present: {sorted(parsed.keys())[:10]}"
                ),
            )
        return usage, None

    if isinstance(parsed, list):
        # Gemini's streamGenerateContent without alt=sse returns a JSON
        # array of partial chunks; usage is on the final chunk(s).
        best_model: Optional[str] = None
        best_prompt = 0
        best_completion = 0
        any_extracted = False
        for chunk in parsed:
            if not isinstance(chunk, dict):
                continue
            u = _parse_chunk(chunk)
            if u is None:
                continue
            any_extracted = True
            best_model = best_model or u.model
            if u.prompt_tokens > best_prompt:
                best_prompt = u.prompt_tokens
            if u.completion_tokens > best_completion:
                best_completion = u.completion_tokens
        last_obj = parsed[-1] if parsed and isinstance(parsed[-1], dict) else {}
        if not any_extracted:
            return (
                None,
                (
                    f"response body is a JSON array of {len(parsed)} items but "
                    "none carried recognized usage/model fields"
                ),
            )
        return (
            ParsedUsage(
                model=best_model,
                prompt_tokens=best_prompt,
                completion_tokens=best_completion,
                response_body=last_obj,
            ),
            None,
        )

    return (
        None,
        (
            f"response body is JSON but neither object nor array "
            f"(got {type(parsed).__name__})"
        ),
    )


def extract_usage_streaming(
    raw_sse: bytes, request_body: dict, request_url: str,
) -> Tuple[Optional[ParsedUsage], Optional[str]]:
    """Parse token usage from accumulated SSE bytes.

    Anthropic splits usage across events (``message_start`` carries
    input, ``message_delta`` carries output), so we scan all frames and
    take the max of each axis.

    Returns ``(usage, parse_failure_reason)``: exactly one is non-None.
    """
    text = raw_sse.decode("utf-8", errors="replace")
    prompt_tokens = 0
    completion_tokens = 0
    model: Optional[str] = request_body.get("model")
    sse_lines_seen = 0
    found_usage = False

    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        sse_lines_seen += 1
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue

        usage = chunk.get("usage") or chunk.get("usageMetadata")
        msg = chunk.get("message")
        if isinstance(msg, dict) and msg.get("usage"):
            usage = usage or msg["usage"]
        if not usage:
            continue

        # Anthropic counts cache_read/creation against input.
        prompt_axis = (
            (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or usage.get("promptTokenCount")
                or 0
            )
            + (usage.get("cache_read_input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
        )
        completion_axis = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or _gemini_completion_tokens(usage)
        )
        if prompt_axis > prompt_tokens:
            prompt_tokens = prompt_axis
        if completion_axis > completion_tokens:
            completion_tokens = completion_axis
        model = model or chunk.get("model") or chunk.get("modelVersion")
        if prompt_axis or completion_axis:
            found_usage = True

    if found_usage:
        return (
            ParsedUsage(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_body={},
            ),
            None,
        )

    # Diagnose.
    if (
        is_openai_compatible_url(request_url)
        and request_body.get("stream") is True
        and not has_include_usage(request_body)
    ):
        reason = (
            "OpenAI-compatible streams omit usage by default — set "
            '`stream_options={"include_usage": True}` on the request '
            "to enable token tracking. (Anthropic streams are unaffected.)"
        )
    elif sse_lines_seen == 0:
        reason = (
            f"streaming response had no `data: ...` SSE frames "
            f"(body length={len(raw_sse)} bytes, "
            f"first 200 bytes={raw_sse[:200]!r}). "
            "If this is a non-SSE chunked stream (e.g. Gemini's "
            "`streamGenerateContent` without `alt=sse`), the non-streaming "
            "code path handles JSON arrays; otherwise this provider's "
            "streaming format may need explicit support in the proxy parser."
        )
    else:
        reason = (
            f"streaming response had {sse_lines_seen} SSE frames "
            "but none carried recognized usage/model fields"
        )
    return None, reason
