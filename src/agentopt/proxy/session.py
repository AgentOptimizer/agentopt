"""Thread-safe in-memory session management for the proxy server."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

from .models import CallRecord


@dataclass
class SessionInfo:
    """State for a single tracking session."""

    session_id: str
    data_id: Optional[str] = None
    combo_id: Optional[str] = None
    agent_id: Optional[str] = None
    records: List[CallRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_record(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)


class SessionManager:
    """Thread-safe in-memory session store.

    Sessions are created by the ``LLMTracker.track()`` context manager and
    ended when the scope exits.  The proxy server looks up the active session
    for each incoming request to attribute ``CallRecord`` instances.
    """

    def __init__(self) -> None:
        self._active: Dict[str, SessionInfo] = {}
        # Ordered dict of archived sessions — preserves insertion order for
        # ``get_all_records``, allows O(1) lookup for late-arriving records
        # from blocking threads that finish after ``end_session``.
        self._ended: Dict[str, SessionInfo] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> SessionInfo:
        """Create and register a new session. Returns the ``SessionInfo``."""
        session_id = f"sess_{uuid4().hex[:12]}"
        session = SessionInfo(
            session_id=session_id,
            data_id=data_id,
            combo_id=combo_id,
            agent_id=agent_id,
        )
        with self._lock:
            self._active[session_id] = session
        return session

    def end_session(self, session_id: str) -> Optional[SessionInfo]:
        """Remove *session_id* from the active set and archive it.

        Returns the ended ``SessionInfo`` (with its accumulated records),
        or ``None`` if the session was not found.
        """
        with self._lock:
            session = self._active.pop(session_id, None)
            if session is not None:
                self._ended[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Look up an active session by ID."""
        with self._lock:
            return self._active.get(session_id)

    def add_record(self, session_id: str, record: CallRecord) -> None:
        """Append a ``CallRecord`` to the session (thread-safe).

        Looks in both active and ended sessions — a blocking upstream
        request may finish after ``end_session`` has moved the session to
        the ended set.  Without this fallback those records would be lost.
        """
        with self._lock:
            session = self._active.get(session_id) or self._ended.get(session_id)
        if session is not None:
            session.add_record(record)

    def get_all_records(self) -> List[CallRecord]:
        """Return every record from all ended sessions."""
        with self._lock:
            ended = list(self._ended.values())
        records: List[CallRecord] = []
        for session in ended:
            with session._lock:
                records.extend(session.records)
        return records

    def force_end_all(self) -> None:
        """Move every active session to the ended list.

        Called during ``LLMTracker.stop()`` to ensure no records are lost.
        """
        with self._lock:
            for sid, session in self._active.items():
                self._ended[sid] = session
            self._active.clear()
