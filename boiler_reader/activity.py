"""Track what the reader is doing right now so the web UI can show it."""

from __future__ import annotations

from threading import Lock

IDLE = "idle"
GRABBING = "grabbing"
ANALYZING = "analyzing"


class ActivityTracker:
    """Thread-safe current phase: idle, grabbing a frame, or analyzing it."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = IDLE
        self._sequence = 0

    def set(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._sequence += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {"phase": self._phase, "sequence": self._sequence}
