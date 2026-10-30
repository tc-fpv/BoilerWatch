"""Shared mutable state used by the web handler, camera callbacks and analysis."""

from __future__ import annotations

from threading import Lock, RLock
from typing import TYPE_CHECKING

from boiler_reader.activity import ActivityTracker
from boiler_reader.alert_delay import AlertDelayTracker
from boiler_reader.decoder import ROI
from boiler_reader.log_buffer import LogBuffer
from boiler_reader.webhook import HomeAssistantWebhookNotifier

if TYPE_CHECKING:
    from boiler_reader.camera import CameraPoller
    from boiler_reader.settings import ViewerSettings


def reader_settings_from(settings: ViewerSettings) -> dict[str, object]:
    """The live, editable settings, seeded from the startup configuration."""
    return {
        "poll_seconds": settings.poll_seconds,
        "low_threshold": settings.low_threshold,
        "high_threshold": settings.high_threshold,
        "alert_delay_minutes": settings.alert_delay_minutes,
        "brightness_threshold": settings.brightness_threshold,
        "expected_format": settings.expected_format,
        "rtsp_url": settings.rtsp_url,
        "rtsp_user": settings.rtsp_user,
        "rtsp_pass": settings.rtsp_pass,
        "ha_webhook_url": settings.ha_webhook_url,
        "ha_webhook_timeout": settings.ha_webhook_timeout,
        "ollama_url": settings.ollama_url,
        "ollama_model": settings.ollama_model,
    }


class AppState:
    """Everything the handler, the camera callbacks and the analyzer share."""

    def __init__(
        self,
        settings: ViewerSettings,
        *,
        camera_poller: CameraPoller | None = None,
        delay_tracker: AlertDelayTracker | None = None,
        notifier: HomeAssistantWebhookNotifier | None = None,
        activity: ActivityTracker | None = None,
    ) -> None:
        self.settings = settings
        self.camera_poller = camera_poller
        self.delay_tracker = delay_tracker or AlertDelayTracker(settings.alert_delay_minutes * 60)
        self.notifier = notifier
        self.activity = activity or ActivityTracker()
        # Live settings edited from the UI; guard every access with settings_lock.
        self.reader_settings = reader_settings_from(settings)
        self.settings_lock = RLock()
        # Serialises webhook delivery and the first on-demand decode.
        self.delivery_lock = Lock()
        self.decode_lock = Lock()
        # The JPEG of the preprocessed image most recently sent to the LLM.
        self.llm_image: bytes | None = None
        # Recent log lines for the UI; attach it to the logging module to fill it.
        self.log_buffer = LogBuffer()
        self._roi: ROI | None = settings.roi
        self._roi_lock = RLock()
        self._cached_reading: dict[str, object] | None = None
        self._cached_lock = RLock()

    def get_roi(self) -> ROI | None:
        with self._roi_lock:
            return self._roi

    def set_roi(self, roi: ROI | None) -> None:
        with self._roi_lock:
            self._roi = roi

    def get_cached_reading(self) -> dict[str, object] | None:
        with self._cached_lock:
            return self._cached_reading

    def set_cached_reading(self, reading: dict[str, object] | None) -> None:
        with self._cached_lock:
            self._cached_reading = reading

    def snapshot_settings(self) -> dict[str, object]:
        with self.settings_lock:
            return dict(self.reader_settings)
