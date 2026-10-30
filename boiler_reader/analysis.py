"""Turn camera frames into readings, alert state and webhook notifications."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Protocol

import cv2
import numpy as np

from boiler_reader.activity import ANALYZING, IDLE, ActivityTracker
from boiler_reader.alert_delay import AlertDelayTracker, DelayStatus
from boiler_reader.camera import CameraPoller
from boiler_reader.decoder import ROI, Reading
from boiler_reader.ocr import LLMUnavailableError
from boiler_reader.reader import DisplayReader, LiveDisplayReader, ReaderState
from boiler_reader.webhook import HomeAssistantWebhookNotifier


# A reading with no confidence at all is retried on a fresh frame; only after this many
# rounds in a row is it left to be declared an error.
MAX_READ_ROUNDS = 3


class AnalyzerState(ReaderState, Protocol):
    """The slice of application state a FrameAnalyzer needs."""

    delay_tracker: AlertDelayTracker
    notifier: HomeAssistantWebhookNotifier | None
    delivery_lock: Lock
    activity: ActivityTracker
    llm_image: bytes | None
    camera_poller: CameraPoller | None

    def set_cached_reading(self, reading: dict[str, object] | None) -> None: ...


def camera_status(image_path: Path) -> dict[str, object]:
    status_path = image_path.with_name("camera-status.json")
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict):
                return status
        except (OSError, json.JSONDecodeError):
            pass
    return {"available": image_path.is_file(), "error": None}


def send_due_webhook(
    notifier: HomeAssistantWebhookNotifier | None,
    delay_tracker: AlertDelayTracker,
    delay: DelayStatus,
    payload: dict[str, object],
    delivery_lock: Lock,
) -> None:
    if notifier is None or not delay.notification_due:
        return
    with delivery_lock:
        if delay_tracker.notification_due() and notifier.send(payload):
            delay_tracker.mark_notified()


def status_for(issue: str | None) -> str:
    return "low" if issue == "low_pressure" else "invalid" if issue == "error" else "ok"


class FrameAnalyzer:
    """Reads frames and keeps the cached reading, alert delay and webhook in step."""

    def __init__(self, state: AnalyzerState, reader: DisplayReader | None = None) -> None:
        self.state = state
        self.reader = reader or LiveDisplayReader(
            state,
            on_located=state.activity.set_roi,
            on_prepared=self.publish_llm_image,
        )

    # -- alerting ---------------------------------------------------------

    def notify(self, issue: str | None, payload: dict[str, object]) -> DelayStatus:
        """Advance the alert delay for this issue and send any webhook now due."""
        state = self.state
        delay = state.delay_tracker.update(issue)
        with state.settings_lock:
            payload["delay_minutes"] = int(state.reader_settings["alert_delay_minutes"])
        send_due_webhook(state.notifier, state.delay_tracker, delay, payload, state.delivery_lock)
        return delay

    def notify_reading(
        self,
        issue: str | None,
        value: object,
        unit: object,
        display_text: object,
        error_code: object,
    ) -> DelayStatus:
        return self.notify(
            issue,
            {
                "status": status_for(issue),
                "issue": issue,
                "pressure": value if unit == "bar" else None,
                "temperature": value if unit == "°C" else None,
                "display": display_text,
                "error_code": error_code,
            },
        )

    def fail(self, status: str, message: str) -> DelayStatus:
        """Record that no reading could be made: alert, and drop the cached reading."""
        delay = self.notify(
            "error",
            {
                "status": status,
                "issue": "error",
                "pressure": None,
                "error": message,
            },
        )
        self.state.set_cached_reading(None)
        return delay

    def issue_for(self, reading: Reading) -> str | None:
        with self.state.settings_lock:
            low = float(self.state.reader_settings["low_threshold"])
            high = float(self.state.reader_settings["high_threshold"])
        if not reading.valid:
            return "error"
        if reading.unit == "°C":
            return None
        if reading.value is not None and reading.value < low:
            return "low_pressure"
        if reading.value is not None and reading.value > high:
            return "error"
        return None

    def build_result(self, reading: Reading, image: np.ndarray, roi: ROI) -> dict[str, object]:
        """Classify a reading, run the alert logic and return the UI payload."""
        issue = self.issue_for(reading)
        delay = self.notify_reading(
            issue, reading.value, reading.unit, reading.display_text, reading.error_code
        )
        return {
            **asdict(reading),
            "roi": asdict(roi),
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "camera": camera_status(self.state.settings.image_path),
            "issue": issue,
            "delay": asdict(delay),
        }

    def refresh_cached(self, cached: dict[str, object]) -> dict[str, object]:
        """Re-run the alert delay for a cached reading so timers advance between polls."""
        issue = cached.get("issue")
        delay = self.notify_reading(
            issue,
            cached.get("value"),
            cached.get("unit"),
            cached.get("display_text"),
            cached.get("error_code"),
        )
        return {
            **cached,
            "camera": camera_status(self.state.settings.image_path),
            "delay": asdict(delay),
        }

    # -- reading ----------------------------------------------------------

    def publish_llm_image(self, processed: np.ndarray) -> None:
        """Store the image being sent to the LLM and tell the UI it changed."""
        ok, encoded = cv2.imencode(".jpg", processed)
        if ok:
            self.state.llm_image = encoded.tobytes()
            self.state.activity.mark_llm_image()

    def _error_result(self, image: np.ndarray, status: str, message: str) -> dict[str, object]:
        """A UI payload for a frame that could not be read, run through the alert delay."""
        delay = self.fail(status, message)
        return {
            "valid": False,
            "value": None,
            "unit": None,
            "raw_pattern": "",
            "confidence": 0.0,
            "display_text": None,
            "error_code": None,
            "error": message,
            "roi": None,
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "camera": camera_status(self.state.settings.image_path),
            "issue": "error",
            "delay": asdict(delay),
        }

    def analyze_on_request(self, image: np.ndarray) -> dict[str, object]:
        """Decode the stored frame for a page request and cache the result."""
        state = self.state
        state.activity.set(ANALYZING)
        try:
            try:
                read = self.reader.read(image)
            except LLMUnavailableError as error:
                result = self._error_result(image, "llm_unavailable", str(error))
            else:
                if read is None:
                    result = self._error_result(
                        image, "invalid", "Display could not be located in frame"
                    )
                else:
                    result = self.build_result(read[0], image, read[1])
            state.set_cached_reading(result)
            return result
        finally:
            state.activity.set(IDLE)

    def analyze_captured(self, frame: np.ndarray | None, error: str | None) -> None:
        """Camera callback: analyze a freshly grabbed frame, or record the failure."""
        state = self.state
        if frame is not None and error is None:
            state.activity.set(ANALYZING)
        try:
            self._analyze_captured(frame, error)
        finally:
            state.activity.set(IDLE)

    def _read_with_retries(
        self, frame: np.ndarray
    ) -> tuple[tuple[Reading, ROI] | None, np.ndarray]:
        """Read the frame, grabbing and reading again while the confidence is 0.0.

        Returns the last read and the frame it came from.  Only called from the camera
        callback, where the poller's capture lock is already held.
        """
        read = self.reader.read(frame)
        for attempt in range(1, MAX_READ_ROUNDS):
            if read is None or read[0].confidence > 0.0:
                break
            poller = self.state.camera_poller
            fresh = poller.recapture() if poller is not None else None
            if fresh is None:
                break
            logging.getLogger("boiler_reader.analysis").info(
                json.dumps({"event": "read_retry", "round": attempt + 1}, separators=(",", ":"))
            )
            frame = fresh
            read = self.reader.read(frame)
        return read, frame

    def _analyze_captured(self, frame: np.ndarray | None, error: str | None) -> None:
        state = self.state
        if frame is None or error is not None:
            self.fail("stale", error or "Camera frame unavailable")
            return
        with state.settings_lock:
            if not str(state.reader_settings.get("ollama_url", "")).strip():
                return
        try:
            read, frame = self._read_with_retries(frame)
        except ValueError:
            self.fail("invalid", "Gauge frame could not be decoded")
            return
        except LLMUnavailableError as error:
            self.fail("llm_unavailable", str(error))
            return
        if read is None:
            self.fail("invalid", "Display could not be located in frame")
            return
        state.set_cached_reading(self.build_result(read[0], frame, read[1]))
