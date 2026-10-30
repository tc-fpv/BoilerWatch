"""Keeps the most recent log lines in memory so the UI can show them."""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from threading import Lock


class LogBuffer(logging.Handler):
    """A logging handler holding the last `capacity` records, each with an increasing id."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__(level=logging.INFO)
        self._entries: deque[dict[str, object]] = deque(maxlen=capacity)
        self._lock = Lock()
        self._last_id = 0

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        with self._lock:
            self._last_id += 1
            self._entries.append({"id": self._last_id, **entry})

    def entries_since(self, last_seen: int = 0) -> dict[str, object]:
        """Entries newer than `last_seen`, plus the id to ask from next time."""
        with self._lock:
            fresh = [entry for entry in self._entries if entry["id"] > last_seen]
            return {"entries": fresh, "last": self._last_id}
