"""Token-usage extraction from LLM API response bodies.

Three input shapes are handled:

* JSON object — OpenAI/Anthropic/Gemini single-response format.
* JSON array — Gemini's ``streamGenerateContent`` non-SSE format
  (returned without ``alt=sse``).
* SSE stream — ``data: {...}`` frames from any provider's streaming API.

Failure mode: the extractors raise :class:`UsageExtractionError` with a
message describing exactly what was searched and what keys were present.
No silent zeros — a successful HTTP call whose usage we can't parse is a
proxy gap that should surface, not a 0-token record.
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


# Recognized token-count keys, in priority order.
_PROMPT_KEYS = ("prompt_tokens", "input_tokens", "promptTokenCount")
_COMPLETION_KEYS = ("completion_tokens", "output_tokens")
_GEMINI_COMPLETION_KEYS = ("candidatesTokenCount", "thoughtsTokenCount")


class UsageExtractionError(ValueError):
    """Raised when token usage can't be parsed from an LLM response.

    The exception message is suitable for attaching to a ``CallRecord.error``
    so a proxy gap is visible without combing through logs.
    """


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
    ``thoughtsTokenCount`` (reasoning) separately; both bill as output.
    Falls back to ``totalTokenCount - promptTokenCount`` when individual
    fields aren't broken out.

    Raises :class:`UsageExtractionError` if no recognized completion field
    is present.
    """
    candidates = usage.get("candidatesTokenCount")
    thoughts = usage.get("thoughtsTokenCount")
    if candidates is not None or thoughts is not None:
        return (candidates or 0) + (thoughts or 0)

    total = usage.get("totalTokenCount")
    prompt = usage.get("promptTokenCount")
    if isinstance(total, int) and isinstance(prompt, int) and total >= prompt:
        return max(0, total - prompt)

    raise UsageExtractionError(
        f"no Gemini completion-token field in usage dict — searched "
        f"{list(_GEMINI_COMPLETION_KEYS)} (or totalTokenCount >= "
        f"promptTokenCount). keys present: {sorted(usage.keys())}"
    )


def _extract_usage_dict(usage: dict) -> Tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from a usage dict.

    Raises :class:`UsageExtractionError` if a prompt-style or completion-style
    key cannot be found.  Both axes must be present — silent zero-fill would
    mask broken responses.
    """
    prompt = next((usage[k] for k in _PROMPT_KEYS if k in usage), None)
    if prompt is None:
        raise UsageExtractionError(
            f"no prompt-token field in usage dict — searched "
            f"{list(_PROMPT_KEYS)}. keys present: {sorted(usage.keys())}"
        )

    completion = next((usage[k] for k in _COMPLETION_KEYS if k in usage), None)
    if completion is None:
        # Gemini's usageMetadata uses different keys.  This raises with
        # its own diagnostic if neither standard nor Gemini fields match.
        completion = _gemini_completion_tokens(usage)

    return prompt, completion


def _parse_chunk(chunk: dict) -> Optional[ParsedUsage]:
    """Extract usage from one response object.

    Returns ``None`` when the chunk has no usage section at all (a normal
    state for, e.g., a Gemini chunk that's only carrying content).  Raises
    :class:`UsageExtractionError` when usage IS present but its fields
    can't be read — that's a real provider-shape mismatch.
    """
    model = chunk.get("model") or chunk.get("modelVersion")
    usage = chunk.get("usage") or chunk.get("usageMetadata")
    msg = chunk.get("message")
    if not usage and isinstance(msg, dict):
        usage = msg.get("usage")
        model = model or msg.get("model")
    if not usage:
        return None
    if not model:
        raise UsageExtractionError(
            f"usage dict present but no `model` / `modelVersion` field. "
            f"chunk keys: {sorted(chunk.keys())}"
        )

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


def extract_usage(body_bytes: bytes) -> ParsedUsage:
    """Parse usage from a non-streaming response body.

    Raises :class:`UsageExtractionError` on any failure — empty body,
    invalid JSON, unrecognized shape, or missing token-count keys.  The
    exception message names exactly what was searched and what was found.
    """
    if not body_bytes:
        raise UsageExtractionError("response body was empty")

    try:
        parsed = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UsageExtractionError(
            f"response body is not valid JSON ({type(exc).__name__}); "
            f"first 200 bytes: {body_bytes[:200]!r}"
        ) from exc

    if isinstance(parsed, dict):
        usage = _parse_chunk(parsed)
        if usage is None:
            raise UsageExtractionError(
                "response body is a JSON object but has no recognized "
                "usage section (expected one of: usage, usageMetadata, "
                "or message.usage). "
                f"keys present: {sorted(parsed.keys())[:10]}"
            )
        return usage

    if isinstance(parsed, list):
        # Gemini's streamGenerateContent without alt=sse returns a JSON
        # array of partial chunks; usage typically lands on the final ones.
        best_model: Optional[str] = None
        best_prompt = 0
        best_completion = 0
        any_extracted = False
        last_error: Optional[UsageExtractionError] = None
        for chunk in parsed:
            if not isinstance(chunk, dict):
                continue
            try:
                u = _parse_chunk(chunk)
            except UsageExtractionError as exc:
                last_error = exc
                continue
            if u is None:
                continue
            any_extracted = True
            best_model = best_model or u.model
            if u.prompt_tokens > best_prompt:
                best_prompt = u.prompt_tokens
            if u.completion_tokens > best_completion:
                best_completion = u.completion_tokens
        if not any_extracted:
            detail = (
                f" (last per-chunk error: {last_error})"
                if last_error is not None
                else ""
            )
            raise UsageExtractionError(
                f"response body is a JSON array of {len(parsed)} items but "
                f"none carried recognized usage/model fields{detail}"
            )
        last_obj = parsed[-1] if parsed and isinstance(parsed[-1], dict) else {}
        return ParsedUsage(
            model=best_model,
            prompt_tokens=best_prompt,
            completion_tokens=best_completion,
            response_body=last_obj,
        )

    raise UsageExtractionError(
        f"response body is JSON but neither object nor array "
        f"(got {type(parsed).__name__})"
    )


def extract_usage_streaming(
    raw_sse: bytes, request_body: dict, request_url: str,
) -> ParsedUsage:
    """Parse token usage from accumulated SSE bytes.

    Anthropic splits usage across events (``message_start`` carries input,
    ``message_delta`` carries output), so we scan all frames and take the
    max of each axis.  Per-frame missing fields are NOT errors — only the
    aggregate result is checked.

    Raises :class:`UsageExtractionError` if no frame yielded usable usage.
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

        # Per-chunk reads stay tolerant: a frame may legitimately carry
        # only one direction (Anthropic message_start has input only).
        # The aggregate is what's checked at the end.
        prompt_axis, completion_axis = _streaming_chunk_axes(usage)
        if prompt_axis > prompt_tokens:
            prompt_tokens = prompt_axis
        if completion_axis > completion_tokens:
            completion_tokens = completion_axis
        model = model or chunk.get("model") or chunk.get("modelVersion")
        if prompt_axis or completion_axis:
            found_usage = True

    if found_usage:
        return ParsedUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_body={},
        )

    # No frame produced usage — diagnose with the most actionable message.
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
    raise UsageExtractionError(reason)


def _streaming_chunk_axes(usage: dict) -> Tuple[int, int]:
    """Lenient per-chunk read for streaming.

    Streaming chunks legitimately carry only one direction at a time, so
    missing keys here fall through to ``0``.  The aggregate ``found_usage``
    check in :func:`extract_usage_streaming` decides whether the *whole*
    response was unparseable.
    """
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
    # Anthropic counts cache_read/creation against input (above).  For
    # completion, prefer explicit fields; fall through to Gemini-style.
    completion_axis = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or (usage.get("candidatesTokenCount") or 0)
        + (usage.get("thoughtsTokenCount") or 0)
    )
    return prompt_axis, completion_axis
