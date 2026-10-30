import json
import threading

import pytest

from boiler_reader.camera_service import CameraService
from boiler_reader.decoder import ROI
from boiler_reader.screenshots import save_mismatch
from boiler_reader.settings import ViewerSettings
from boiler_reader.settings_service import SettingsService
from boiler_reader.state import AppState


class FakePoller:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.captures = 0
        self.ran = threading.Event()
        FakePoller.instances.append(self)

    def capture_once(self) -> bool:
        self.captures += 1
        return True

    def run(self, stop_event, capture_on_start=True) -> None:
        if capture_on_start:
            self.capture_once()
        self.ran.set()
        stop_event.wait(5)


def test_camera_service_restart_stops_the_old_poller_and_publishes_the_new_one(tmp_path) -> None:
    FakePoller.instances.clear()
    published, grab_states = [], []
    service = CameraService(
        tmp_path / "last.jpg",
        on_frame=lambda frame, error: None,
        on_grab_state=grab_states.append,
        on_poller=published.append,
        poller_factory=FakePoller,
    )

    service.start("rtsp://a", "", "", 10, first_capture_in_background=True)
    first = published[0]
    assert first.ran.wait(2) and first.captures == 1 and grab_states == [True]

    service.start("rtsp://b", "u", "p", 20)
    second = published[1]
    assert second.captures == 1 and second.kwargs["rtsp_url"] == "rtsp://b"
    service.stop()


def _state(tmp_path):
    settings = ViewerSettings(
        image_path=tmp_path / "last.jpg",
        roi=ROI(0, 0, 10, 10),
        settings_state_path=tmp_path / "reader-settings.json",
    )
    return AppState(settings)


def test_reader_settings_update_applies_persists_and_clears_the_cache(tmp_path) -> None:
    state = _state(tmp_path)
    state.set_cached_reading({"valid": True})

    applied = SettingsService(state).update_reader(
        {"poll_seconds": 12, "low_threshold": 1.2, "high_threshold": 1.8, "alert_delay_minutes": 3}
    )

    assert applied["poll_seconds"] == 12
    assert state.snapshot_settings()["low_threshold"] == 1.2
    assert state.delay_tracker.delay_seconds == 180
    assert state.get_cached_reading() is None
    assert json.loads((tmp_path / "reader-settings.json").read_text())["poll_seconds"] == 12


def test_reader_settings_update_rejects_fractional_poll_seconds(tmp_path) -> None:
    with pytest.raises(ValueError, match="whole number"):
        SettingsService(_state(tmp_path)).update_reader(
            {"poll_seconds": 1.5, "low_threshold": 1.0, "high_threshold": 1.5, "alert_delay_minutes": 5}
        )


def test_admin_update_restarts_the_camera_only_when_an_rtsp_url_is_set(tmp_path) -> None:
    restarts = []
    service = SettingsService(_state(tmp_path), restart_camera=lambda *args: restarts.append(args))

    service.update_admin({"rtsp_url": "", "poll_seconds": 30})
    service.update_admin({"rtsp_url": "rtsp://cam", "rtsp_user": "u", "poll_seconds": 15})

    assert restarts == [("rtsp://cam", "u", "", 15)]


def test_save_mismatch_validates_then_reports_a_missing_frame(tmp_path) -> None:
    settings = _state(tmp_path).settings
    with pytest.raises(ValueError, match="unit"):
        save_mismatch(settings, None, {"value": "12", "unit": "psi"})
    assert save_mismatch(settings, None, {"value": "12", "unit": "bar"}) is None


def test_log_buffer_returns_only_entries_after_the_given_id() -> None:
    import logging

    from boiler_reader.log_buffer import LogBuffer

    buffer = LogBuffer(capacity=2)
    logger = logging.getLogger("test_log_buffer")
    logger.setLevel(logging.INFO)
    logger.addHandler(buffer)
    try:
        for text in ("one", "two", "three"):
            logger.info(text)
    finally:
        logger.removeHandler(buffer)

    everything = buffer.entries_since(0)
    newer = buffer.entries_since(2)

    assert [e["message"] for e in everything["entries"]] == ["two", "three"]
    assert everything["last"] == 3
    assert [e["message"] for e in newer["entries"]] == ["three"]
