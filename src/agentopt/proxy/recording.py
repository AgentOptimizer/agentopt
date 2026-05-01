"""Build and dispatch ``CallRecord`` instances from raw upstream results.

Both interception paths — the in-process httpx wrapper and the mitmproxy
addon — reduce to the same five inputs: a session, the request body, the
response body, an upstream URL, a latency measurement, and a status. This
module owns the conversion to ``CallRecord`` so the two paths can't drift.

The token-extraction-failure surfacing rules (``<parse-failed>`` sentinel,
warn-once-per-host) live here too, because they're properties of the
recording semantics, not of either transport.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Set
from urllib.parse import urlparse

from .models import CallRecord
from .session import SessionInfo, SessionManager
from .usage import (
    PARSE_FAILED_MODEL,
    UsageExtractionError,
    extract_usage,
    extract_usage_streaming,
)

logger = logging.getLogger(__name__)


class Recorder:
    """Builds ``CallRecord`` instances and hands them to a ``SessionManager``.

    Holds the warn-once-per-host set so we don't log-spam token-extraction
    failures across many requests to the same upstream.  Thread-safe — both
    the main thread (in-process httpx wrapper) and any number of mitmproxy
    addon threads can call ``record`` concurrently.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        self._sm = session_manager
        self._stream_warned_hosts: Set[str] = set()
        self._lock = threading.Lock()

    def record(
        self,
        session: SessionInfo,
        request_body_json: dict,
        response_body: Optional[bytes],
        request_url: str,
        latency_seconds: float,
        *,
        cached: bool,
        is_streaming: bool,
        status_code: int = 200,
        error: Optional[str] = None,
    ) -> None:
        """Build a ``CallRecord`` and add it to the session.

        ``response_body`` is ``None`` only on the connection-failure path,
        where the caller already supplied an ``error`` message.  On any
        HTTP response (2xx, 4xx, 5xx) ``response_body`` is the raw bytes
        and we still record — non-2xx responses are real work that should
        appear in the ledger.
        """
        usage = None
        parse_err: Optional[str] = None

        if response_body is not None:
            try:
                if is_streaming and status_code == 200:
                    usage = extract_usage_streaming(
                        response_body, request_body_json, request_url,
                    )
                else:
                    usage = extract_usage(response_body)
            except UsageExtractionError as exc:
                parse_err = str(exc)

        model: Optional[str] = request_body_json.get("model")
        prompt_tokens = 0
        completion_tokens = 0
        resp_body_for_record: dict = {}
        if usage is not None:
            model = model or usage.model
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            resp_body_for_record = usage.response_body

        record_error = error
        # Surface token-extraction failures on otherwise-successful calls
        # loudly — silent zeros mask real proxy gaps.
        if usage is None and status_code == 200 and not error and parse_err is not None:
            record_error = (
                f"agentopt proxy could not extract token usage: {parse_err}. "
                f"URL={request_url}"
            )
            self._maybe_warn(request_url, parse_err, model)
            if not model:
                model = PARSE_FAILED_MODEL
        elif not model:
            model = "unknown"

        self._sm.add_record(
            session.session_id,
            CallRecord(
                data_id=session.data_id,
                combo_id=session.combo_id,
                agent_id=session.agent_id,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=latency_seconds,
                request_url=request_url,
                request_body=request_body_json,
                response_body=resp_body_for_record,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cached=cached,
                status_code=status_code,
                error=record_error,
            ),
        )

    def _maybe_warn(self, request_url: str, reason: str, model: Optional[str],) -> None:
        host = urlparse(request_url).hostname or request_url
        with self._lock:
            if host in self._stream_warned_hosts:
                return
            self._stream_warned_hosts.add(host)
        logger.warning(
            "agentopt: %s call to %s succeeded but token usage could not be "
            "extracted — %s",
            model or "<unknown>",
            request_url,
            reason,
        )
