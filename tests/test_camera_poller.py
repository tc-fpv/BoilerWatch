import cv2
import numpy as np
import json
import pytest

from boiler_reader.camera import CameraPoller


class StopAfterWaits:
    def __init__(self, timeline: list[tuple[str, float | None]], stop_after: int) -> None:
        self.timeline = timeline
        self.stop_after = stop_after
        self.wait_count = 0
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def wait(self, timeout: float) -> bool:
        self.timeline.append(("wait", timeout))
        self.wait_count += 1
        if self.wait_count >= self.stop_after:
            self.stopped = True
            return True
        return False


def test_captures_immediately_then_waits_poll_interval(tmp_path) -> None:
    timeline: list[tuple[str, float | None]] = []
    frame = np.full((24, 32, 3), 127, dtype=np.uint8)

    def capture_frame() -> np.ndarray:
        timeline.append(("capture", None))
        return frame

    poller = CameraPoller(
        rtsp_url="rtsp://camera.invalid/stream",
        username="viewer",
        password="not-logged",
        image_path=tmp_path / "latest.jpg",
        poll_seconds=17,
        capture_fn=capture_frame,
        logger=lambda event, **fields: None,
    )

    stop_event = StopAfterWaits(timeline, stop_after=2)
    poller.run_with_wait(stop_event, capture_on_start=True, wait_fn=stop_event.wait)

    # Polling is start-to-start, so each wait is the interval minus the capture time.
    assert [event for event, _ in timeline] == ["capture", "wait", "capture", "wait"]
    assert [value for event, value in timeline if event == "wait"] == [
        pytest.approx(17, abs=1),
        pytest.approx(17, abs=1),
    ]
    saved = cv2.imread(str(tmp_path / "latest.jpg"))
    assert saved.shape == frame.shape


def test_failed_capture_removes_previous_frame_and_records_error(tmp_path) -> None:
    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    frames = iter((frame, None))
    poller = CameraPoller(
        rtsp_url="rtsp://camera.invalid/stream",
        username="viewer",
        password="secret-value",
        image_path=tmp_path / "latest.jpg",
        poll_seconds=1,
        capture_fn=lambda: next(frames),
        logger=lambda event, **fields: None,
    )

    stop_event = StopAfterWaits([], stop_after=2)
    poller.run_with_wait(stop_event, capture_on_start=True, wait_fn=stop_event.wait)

    assert not (tmp_path / "latest.jpg").exists()
    status = json.loads((tmp_path / "camera-status.json").read_text(encoding="utf-8"))
    assert not status["available"]
    assert "did not return a frame" in status["error"]
    assert "secret-value" not in json.dumps(status)


def test_frame_callback_receives_success_and_failure(tmp_path) -> None:
    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    frames = iter((frame, None))
    updates: list[tuple[bool, str | None]] = []
    poller = CameraPoller(
        rtsp_url="rtsp://camera.invalid/stream",
        username="",
        password="",
        image_path=tmp_path / "latest.jpg",
        poll_seconds=1,
        capture_fn=lambda: next(frames),
        logger=lambda event, **fields: None,
        on_frame=lambda captured, error: updates.append((captured is not None, error)),
    )

    assert poller.capture_once()
    assert not poller.capture_once()

    assert updates == [
        (True, None),
        (False, "RTSP camera did not return a frame"),
    ]

def _poller(tmp_path, capture_fn, events=None):
    return CameraPoller(
        rtsp_url="rtsp://camera.invalid/stream",
        username="viewer",
        password="secret-value",
        image_path=tmp_path / "latest.jpg",
        poll_seconds=1,
        capture_fn=capture_fn,
        logger=(lambda event, **fields: events.append(event)) if events is not None else (lambda event, **fields: None),
    )


def test_recapture_saves_and_returns_the_new_frame(tmp_path) -> None:
    frame = np.full((24, 32, 3), 200, dtype=np.uint8)
    poller = _poller(tmp_path, lambda: frame)

    result = poller.recapture()

    assert result is frame
    assert (tmp_path / "latest.jpg").is_file()
    status = json.loads((tmp_path / "camera-status.json").read_text(encoding="utf-8"))
    assert status["available"] is True


def test_failed_recapture_returns_none_and_keeps_the_previous_frame(tmp_path) -> None:
    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    frames = iter((frame, None))
    events: list[str] = []
    poller = _poller(tmp_path, lambda: next(frames), events)
    poller.capture_once()
    saved = (tmp_path / "latest.jpg").read_bytes()

    result = poller.recapture()

    assert result is None
    assert (tmp_path / "latest.jpg").read_bytes() == saved
    assert "camera_regrab_failed" in events
