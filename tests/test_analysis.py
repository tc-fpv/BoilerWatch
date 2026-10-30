from pathlib import Path

import cv2

from boiler_reader.activity import IDLE
from boiler_reader.alert_delay import AlertDelayTracker
from boiler_reader.analysis import FrameAnalyzer
from boiler_reader.decoder import DigitBox, ROI
from boiler_reader.settings import ViewerSettings
from boiler_reader.state import AppState
from boiler_reader.webhook import HomeAssistantWebhookNotifier

FIXTURE = Path(__file__).with_name("boiler.PNG")


def _state(tmp_path, **overrides):
    requests = []
    notifier = HomeAssistantWebhookNotifier(
        "http://ha.local/api/webhook/test-id",
        post_json=lambda url, payload, timeout: requests.append(payload) or 200,
    )
    settings = ViewerSettings(
        image_path=tmp_path / "last.jpg",
        roi=ROI(330, 238, 110, 94),
        digit_boxes=(DigitBox(0, 0, 22, 85), DigitBox(30, 3, 48, 76)),
        decimal_box=DigitBox(13, 68, 9, 8),
        low_threshold=1.4,
        alert_delay_minutes=1,
        **overrides,
    )
    clock = [0.0]
    tracker = AlertDelayTracker(10, clock=lambda: clock[0])
    return AppState(settings, delay_tracker=tracker, notifier=notifier), requests, clock


def test_camera_failure_clears_cached_reading_and_reports_stale(tmp_path) -> None:
    state, requests, clock = _state(tmp_path)
    state.set_cached_reading({"valid": True})
    analyzer = FrameAnalyzer(state)

    analyzer.analyze_captured(None, "RTSP camera did not return a frame")
    assert state.get_cached_reading() is None
    assert state.activity.snapshot()["phase"] == IDLE
    assert not requests

    clock[0] = 10.0
    analyzer.analyze_captured(None, "RTSP camera did not return a frame")
    assert requests[0]["status"] == "stale"
    assert requests[0]["error"] == "RTSP camera did not return a frame"


def test_captured_frame_is_skipped_without_an_llm_url(tmp_path) -> None:
    state, requests, _ = _state(tmp_path)
    frame = cv2.imread(str(FIXTURE))

    FrameAnalyzer(state).analyze_captured(frame, None)

    assert state.get_cached_reading() is None
    assert state.activity.snapshot()["phase"] == IDLE


def test_on_request_analysis_caches_the_reading_and_flags_low_pressure(tmp_path) -> None:
    state, _, _ = _state(tmp_path)
    image = cv2.imread(str(FIXTURE))

    result = FrameAnalyzer(state).analyze_on_request(image)

    assert state.get_cached_reading() is result
    assert result["issue"] == "low_pressure"
    assert result["roi"] == {"x": 330, "y": 238, "width": 110, "height": 94}
    assert state.activity.snapshot()["phase"] == IDLE


def test_refresh_cached_advances_the_delay_and_sends_the_webhook_once(tmp_path) -> None:
    state, requests, clock = _state(tmp_path)
    analyzer = FrameAnalyzer(state)
    cached = analyzer.analyze_on_request(cv2.imread(str(FIXTURE)))
    assert not requests

    clock[0] = 10.0
    due = analyzer.refresh_cached(cached)
    repeated = analyzer.refresh_cached(cached)

    assert due["delay"]["notification_due"]
    assert not repeated["delay"]["notification_due"]
    assert len(requests) == 1
    assert requests[0]["status"] == "low"


def test_unreachable_llm_alerts_as_llm_unavailable_not_a_display_fault(tmp_path) -> None:
    from boiler_reader.ocr import LLMUnavailableError

    class UnreachableReader:
        def read(self, image):
            raise LLMUnavailableError("No reply from the LLM server")

    state, requests, clock = _state(tmp_path)
    analyzer = FrameAnalyzer(state, reader=UnreachableReader())
    image = cv2.imread(str(FIXTURE))

    first = analyzer.analyze_on_request(image)
    clock[0] = 10.0
    analyzer.analyze_on_request(image)

    assert first["issue"] == "error" and first["error"] == "No reply from the LLM server"
    assert requests[0]["status"] == "llm_unavailable"
    assert len(requests) == 1


class _ScriptedReader:
    """Returns the scripted confidences in order, then repeats the last."""

    def __init__(self, confidences):
        self.confidences = list(confidences)
        self.reads = 0

    def read(self, image):
        from boiler_reader.decoder import ROI, Reading

        confidence = self.confidences[min(self.reads, len(self.confidences) - 1)]
        self.reads += 1
        reading = Reading(
            value=1.2 if confidence else None,
            valid=bool(confidence),
            raw_pattern="12",
            confidence=confidence,
            display_text="12",
            error_code=None,
            unit="bar" if confidence else None,
        )
        return reading, ROI(0, 0, 10, 10)


class _FakePoller:
    def __init__(self, frame):
        self.frame = frame
        self.regrabs = 0

    def recapture(self):
        self.regrabs += 1
        return self.frame


def _analyze_with(tmp_path, confidences):
    state, requests, clock = _state(tmp_path)
    state.reader_settings["ollama_url"] = "http://llm"
    image = cv2.imread(str(FIXTURE))
    state.camera_poller = _FakePoller(image)
    reader = _ScriptedReader(confidences)
    analyzer = FrameAnalyzer(state, reader=reader)
    analyzer.analyze_captured(image, None)
    return state, reader, state.camera_poller


def test_zero_confidence_regrabs_and_reads_again(tmp_path) -> None:
    state, reader, poller = _analyze_with(tmp_path, [0.0, 1.0])

    assert reader.reads == 2 and poller.regrabs == 1
    assert state.get_cached_reading()["valid"] is True


def test_three_zero_confidence_rounds_are_declared_an_error(tmp_path) -> None:
    state, reader, poller = _analyze_with(tmp_path, [0.0])

    assert reader.reads == 3 and poller.regrabs == 2
    assert state.get_cached_reading()["issue"] == "error"


def test_nonzero_confidence_is_not_retried(tmp_path) -> None:
    state, reader, poller = _analyze_with(tmp_path, [0.5])

    assert reader.reads == 1 and poller.regrabs == 0


def test_failed_regrab_stops_retrying_and_uses_the_last_reading(tmp_path) -> None:
    state, requests, clock = _state(tmp_path)
    state.reader_settings["ollama_url"] = "http://llm"
    image = cv2.imread(str(FIXTURE))
    poller = _FakePoller(None)
    state.camera_poller = poller
    reader = _ScriptedReader([0.0])

    FrameAnalyzer(state, reader=reader).analyze_captured(image, None)

    assert reader.reads == 1 and poller.regrabs == 1
    assert state.get_cached_reading()["issue"] == "error"
