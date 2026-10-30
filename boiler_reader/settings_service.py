"""Applying settings edited from the UI: validate, persist, and update live components."""

from __future__ import annotations

import math
from typing import Callable

from boiler_reader.settings import (
    SETTINGS_KEYS,
    persist_json,
    validate_admin_settings,
    validate_reader_settings,
)
from boiler_reader.state import AppState
from boiler_reader.webhook import HomeAssistantWebhookNotifier

# Called as restart_camera(rtsp_url, rtsp_user, rtsp_pass, poll_seconds).
RestartCamera = Callable[[str, str, str, int], None]
NotifierFactory = Callable[[dict[str, object]], HomeAssistantWebhookNotifier | None]


def make_notifier(s: dict[str, object]) -> HomeAssistantWebhookNotifier | None:
    url = str(s.get("ha_webhook_url", "")).strip()
    if not url:
        return None
    return HomeAssistantWebhookNotifier(
        url,
        timeout_seconds=int(s.get("ha_webhook_timeout", 5)),
    )


def _whole_number(raw: object, name: str) -> int:
    value = float(raw)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(value)


class SettingsService:
    """Raises ValueError/KeyError/TypeError for bad input and OSError if saving fails."""

    def __init__(
        self,
        state: AppState,
        restart_camera: RestartCamera | None = None,
        notifier_factory: NotifierFactory = make_notifier,
    ) -> None:
        self._state = state
        self._restart_camera = restart_camera
        self._notifier_factory = notifier_factory

    def update_reader(self, payload: dict[str, object]) -> dict[str, object]:
        """Apply the thresholds and timing settings; returns the values applied."""
        state = self._state
        poll_seconds = _whole_number(payload["poll_seconds"], "POLL_SECONDS")
        low_threshold = float(payload["low_threshold"])
        high_threshold = float(payload["high_threshold"])
        alert_delay_minutes = _whole_number(payload["alert_delay_minutes"], "ALERT_DELAY_MINUTES")
        validate_reader_settings(poll_seconds, low_threshold, high_threshold, alert_delay_minutes)
        applied = {
            "poll_seconds": poll_seconds,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "alert_delay_minutes": alert_delay_minutes,
        }
        self._store(applied)
        if state.camera_poller is not None:
            state.camera_poller.set_poll_seconds(poll_seconds)
        state.delay_tracker.set_delay_seconds(alert_delay_minutes * 60)
        return applied

    def update_admin(self, payload: dict[str, object]) -> None:
        """Apply the full admin settings, restarting the camera if an RTSP URL is set."""
        state = self._state
        saved: dict[str, object] = {
            "rtsp_url": str(payload.get("rtsp_url", "")).strip(),
            "rtsp_user": str(payload.get("rtsp_user", "")).strip(),
            "rtsp_pass": str(payload.get("rtsp_pass", "")),
            "poll_seconds": int(float(payload.get("poll_seconds", 30))),
            "ha_webhook_url": str(payload.get("ha_webhook_url", "")).strip(),
            "ha_webhook_timeout": int(payload.get("ha_webhook_timeout", 5)),
            "low_threshold": float(payload.get("low_threshold", 1.0)),
            "high_threshold": float(payload.get("high_threshold", 1.5)),
            "alert_delay_minutes": int(float(payload.get("alert_delay_minutes", 5))),
            "brightness_threshold": int(payload.get("brightness_threshold", 30)),
            "expected_format": str(payload.get("expected_format", "auto")),
            "ollama_url": str(payload.get("ollama_url", "")).strip(),
            "ollama_model": str(payload.get("ollama_model", "lfm3-vl:latest")).strip(),
        }
        validate_admin_settings(saved)
        self._store(saved)
        state.notifier = self._notifier_factory(saved)
        state.delay_tracker.set_delay_seconds(int(saved["alert_delay_minutes"]) * 60)
        if state.camera_poller is not None:
            state.camera_poller.set_poll_seconds(int(saved["poll_seconds"]))
        if self._restart_camera is not None and saved["rtsp_url"]:
            self._restart_camera(
                str(saved["rtsp_url"]),
                str(saved["rtsp_user"]),
                str(saved["rtsp_pass"]),
                int(saved["poll_seconds"]),
            )

    def _store(self, changes: dict[str, object]) -> None:
        """Merge into the live settings, save them, and drop the now-stale cached reading."""
        state = self._state
        with state.settings_lock:
            state.reader_settings.update(changes)
            full = {k: state.reader_settings[k] for k in SETTINGS_KEYS if k in state.reader_settings}
        if state.settings.settings_state_path is not None:
            persist_json(state.settings.settings_state_path, full)
        state.set_cached_reading(None)
