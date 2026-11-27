"""Data models for agentopt.proxy."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CallRecord:
    """Record of a single intercepted LLM API call."""

    # Attribution (from ContextVars)
    data_id: Optional[str]
    combo_id: Optional[str]
    agent_id: Optional[str]

    # LLM call metrics
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float

    # Full fidelity
    request_url: str
    request_body: Dict[str, Any] = field(default_factory=dict)
    response_body: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    cached: bool = False

    # Outcome — 200 on success; upstream HTTP status or 0 on connection failure.
    status_code: int = 200
    # Populated when the upstream call failed (non-2xx, connection error, etc.).
    error: Optional[str] = None
    # Known pre-send failures cannot incur an upstream charge.
    request_not_sent: bool = False

    @property
    def transport_charge_unknown(self) -> bool:
        return self.status_code == 0 and not self.request_not_sent

    @property
    def usage_unavailable(self) -> bool:
        """A successful response whose token usage the recorder could not measure."""
        return self.status_code == 200 and self.error is not None
