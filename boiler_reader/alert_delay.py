"""Track how long a low-pressure or error condition remains active."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Callable


@dataclass(frozen=True)
class DelayStatus:
    issue: str | None
    elapsed_seconds: int
    remaining_seconds: int
    eligible: bool
    notification_due: bool = False


class AlertDelayTracker:
    def __init__(
        self,
        delay_seconds: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if delay_seconds < 1:
            raise ValueError("delay_seconds must be positive")
        self.delay_seconds = delay_seconds
        self._started_at: float | None = None
        self._notification_sent = False
        self._lock = RLock()
        self._clock = clock

    def update(self, issue: str | None, now: float | None = None) -> DelayStatus:
        if issue not in (None, "low_pressure", "error"):
            raise ValueError("issue must be None, 'low_pressure', or 'error'")
        current_time = self._clock() if now is None else now
        with self._lock:
            if issue is None:
                self._started_at = None
                self._notification_sent = False
                return DelayStatus(None, 0, self.delay_seconds, False)
            if self._started_at is None:
                self._started_at = current_time

            elapsed_seconds = max(0, int(current_time - self._started_at))
            remaining_seconds = max(0, self.delay_seconds - elapsed_seconds)
            eligible = remaining_seconds == 0
            return DelayStatus(
                issue=issue,
                elapsed_seconds=elapsed_seconds,
                remaining_seconds=remaining_seconds,
                eligible=eligible,
                notification_due=eligible and not self._notification_sent,
            )

    def mark_notified(self) -> None:
        with self._lock:
            if self._started_at is not None:
                self._notification_sent = True

    def notification_due(self) -> bool:
        with self._lock:
            if self._started_at is None or self._notification_sent:
                return False
            return self._clock() - self._started_at >= self.delay_seconds

    def set_delay_seconds(self, delay_seconds: int) -> None:
        if delay_seconds < 1:
            raise ValueError("delay_seconds must be positive")
        with self._lock:
            self.delay_seconds = delay_seconds