import json
from pathlib import Path
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from boiler_reader.decoder import DigitBox, ROI
from boiler_reader.camera import CameraPoller
from boiler_reader.alert_delay import AlertDelayTracker
from boiler_reader.image_viewer import ViewerSettings, annotate_roi, make_handler, settings_from_env
from boiler_reader.webhook import HomeAssistantWebhookNotifier


def test_annotate_roi_marks_boundary_without_mutating_source() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)

    annotated = annotate_roi(image, ROI(10, 8, 20, 15))

    assert np.array_equal(image, np.zeros_like(image))
    assert tuple(annotated[8, 10]) == (0, 100, 255)
    assert tuple(annotated[15, 15]) == (0, 0, 0)


def test_settings_read_port_and_image_path_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WEB_PORT", "8181")
    monkeypatch.setenv("LAST_IMAGE_PATH", str(tmp_path / "frame.jpg"))
    monkeypatch.setenv("ROI_X", "12")
    monkeypatch.setenv("ROI_Y", "34")
    monkeypatch.setenv("ROI_W", "56")
    monkeypatch.setenv("ROI_H", "78")
    monkeypatch.setenv("DIGIT_BOXES", "1,2,30,40;50,60,70,80")
    monkeypatch.setenv("DECIMAL_BOX", "90,100,10,11")

    settings = settings_from_env()

    assert settings.port == 8181
    assert settings.image_path == tmp_path / "frame.jpg"
    assert settings.roi == ROI(12, 34, 56, 78)
    assert settings.digit_boxes == (DigitBox(1, 2, 30, 40), DigitBox(50, 60, 70, 80))
    assert settings.decimal_box == DigitBox(90, 100, 10, 11)


def test_settings_reject_invalid_port(monkeypatch) -> None:
    monkeypatch.setenv("WEB_PORT", "70000")

    with pytest.raises(ValueError, match="WEB_PORT"):
        settings_from_env()


def test_poll_and_low_limit_load_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("POLL_SECONDS", "17")
    monkeypatch.setenv("LOW_THRESHOLD", "1.4")
    monkeypatch.setenv("ALERT_DELAY_MINUTES", "8")
    monkeypatch.setenv("READER_SETTINGS_PATH", str(tmp_path / "reader-settings.json"))

    settings = settings_from_env()

    assert settings.poll_seconds == 17
    assert settings.low_threshold == 1.4
    assert settings.alert_delay_minutes == 8


def test_poll_and_low_limit_load_from_dotenv(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text(
        "POLL_SECONDS=19\nLOW_THRESHOLD=1.3\nALERT_DELAY_MINUTES=7\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAST_IMAGE_PATH", str(tmp_path / "last.jpg"))
    monkeypatch.delenv("POLL_SECONDS", raising=False)
    monkeypatch.delenv("LOW_THRESHOLD", raising=False)

    settings = settings_from_env()

    assert settings.poll_seconds == 19
    assert settings.low_threshold == 1.3
    assert settings.alert_delay_minutes == 7


def test_saved_poll_and_low_limit_override_environment(monkeypatch, tmp_path) -> None:
    settings_path = tmp_path / "reader-settings.json"
    settings_path.write_text(
        json.dumps({"poll_seconds": 45, "low_threshold": 1.2, "alert_delay_minutes": 11}),
        encoding="utf-8",
    )
    monkeypatch.setenv("POLL_SECONDS", "17")
    monkeypatch.setenv("LOW_THRESHOLD", "1.4")
    monkeypatch.setenv("ALERT_DELAY_MINUTES", "8")
    monkeypatch.setenv("READER_SETTINGS_PATH", str(settings_path))

    settings = settings_from_env()

    assert settings.poll_seconds == 45
    assert settings.low_threshold == 1.2
    assert settings.alert_delay_minutes == 11


def test_viewer_serves_latest_image_with_roi(tmp_path) -> None:
    image_path = Path(__file__).with_name("boiler.PNG")
    settings = ViewerSettings(
        image_path=image_path,
        roi=ROI(330, 238, 110, 94),
        digit_boxes=(DigitBox(0, 0, 22, 85), DigitBox(30, 3, 48, 76)),
        decimal_box=DigitBox(13, 68, 9, 8),
        roi_state_path=tmp_path / "roi.json",
        settings_state_path=tmp_path / "reader-settings.json",
    )
    poller = CameraPoller(
        rtsp_url="rtsp://camera.invalid/stream",
        username="",
        password="",
        image_path=image_path,
        poll_seconds=30,
        capture_fn=lambda: None,
        logger=lambda event, **fields: None,
    )
    delay_tracker = AlertDelayTracker(300)
    handler = make_handler(settings, poller, delay_tracker)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = urlopen(f"http://127.0.0.1:{server.server_port}/")
        page = root.read().decode("utf-8")
        assert '<img id="frame" src="/last.jpg"' in page
        assert "GAUGE DECODER" in page
        assert 'id="error-code"' in page
        assert "fetch('/reading.json" in page
        assert 'id="poll-seconds"' in page
        assert 'id="low-threshold"' in page
        assert 'id="low-threshold" type="number" min="0" max="10" step="0.1" value="1.0"' in page
        assert 'id="alert-delay-minutes"' in page
        assert 'id="issue-delay"' in page
        assert 'id="capture-now"' in page
        assert 'id="send-test-webhook"' in page
        assert 'id="last-grabbed"' in page
        assert "Save settings" in page

        favicon = urlopen(f"http://127.0.0.1:{server.server_port}/favicon.ico")
        assert favicon.status == 204

        response = urlopen(f"http://127.0.0.1:{server.server_port}/last.jpg?t=123")
        assert response.headers.get_content_type() == "image/jpeg"
        frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert frame.shape == (480, 640, 3)
        assert frame[238, 330, 2] > frame[238, 330, 0]

        reading_response = urlopen(f"http://127.0.0.1:{server.server_port}/reading.json?t=123")
        assert reading_response.headers.get_content_type() == "application/json"
        reading = json.loads(reading_response.read())
        assert reading["valid"]
        assert reading["value"] == 1.3
        assert reading["display_text"] == "13"

        settings_response = urlopen(f"http://127.0.0.1:{server.server_port}/settings.json")
        initial_settings = json.loads(settings_response.read())
        assert initial_settings["poll_seconds"] == 30
        assert initial_settings["low_threshold"] == 1.0
        assert initial_settings["alert_delay_minutes"] == 5

        save_settings = Request(
            f"http://127.0.0.1:{server.server_port}/settings",
            data=json.dumps({
                "poll_seconds": 17,
                "low_threshold": 1.4,
                "alert_delay_minutes": 7,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        saved_response = urlopen(save_settings)
        saved_settings = json.loads(saved_response.read())
        assert saved_settings == {
            "poll_seconds": 17,
            "low_threshold": 1.4,
            "alert_delay_minutes": 7,
        }
        assert poller.get_poll_seconds() == 17
        assert delay_tracker.delay_seconds == 420
        assert json.loads(settings.settings_state_path.read_text(encoding="utf-8")) == saved_settings

        low_reading_response = urlopen(
            f"http://127.0.0.1:{server.server_port}/reading.json"
        )
        low_reading = json.loads(low_reading_response.read())
        assert low_reading["issue"] == "low_pressure"
        assert low_reading["delay"]["issue"] == "low_pressure"
        assert 0 < low_reading["delay"]["remaining_seconds"] <= 420

        updated_roi = ROI(330, 238, 112, 96)
        request = Request(
            f"http://127.0.0.1:{server.server_port}/roi",
            data=json.dumps({
                "x": updated_roi.x,
                "y": updated_roi.y,
                "width": updated_roi.width,
                "height": updated_roi.height,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        update_response = urlopen(request)
        updated_reading = json.loads(update_response.read())
        assert updated_reading["valid"]
        assert updated_reading["value"] == 1.3
        assert updated_reading["roi"] == {
            "x": 330,
            "y": 238,
            "width": 112,
            "height": 96,
        }
        assert json.loads(settings.roi_state_path.read_text(encoding="utf-8")) == updated_reading["roi"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_missing_camera_frame_returns_recorded_capture_error(tmp_path) -> None:
    image_path = tmp_path / "last.jpg"
    (tmp_path / "camera-status.json").write_text(
        json.dumps({"available": False, "error": "RTSP camera did not return a frame"}),
        encoding="utf-8",
    )
    settings = ViewerSettings(
        image_path=image_path,
        roi=ROI(0, 0, 50, 50),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{server.server_port}/reading.json")
        assert error.value.code == 503
        payload = json.loads(error.value.read())
        assert payload["error"] == "RTSP camera did not return a frame"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_webhook_fires_once_when_low_pressure_delay_expires() -> None:
    current_time = [0.0]
    requests = []
    responses = iter((503, 200))
    tracker = AlertDelayTracker(10, clock=lambda: current_time[0])
    notifier = HomeAssistantWebhookNotifier(
        "http://ha.local/api/webhook/test-id",
        post_json=lambda url, payload, timeout: requests.append(payload) or next(responses),
    )
    settings = ViewerSettings(
        image_path=Path(__file__).with_name("boiler.PNG"),
        roi=ROI(330, 238, 110, 94),
        digit_boxes=(DigitBox(0, 0, 22, 85), DigitBox(30, 3, 48, 76)),
        decimal_box=DigitBox(13, 68, 9, 8),
        low_threshold=1.4,
        alert_delay_minutes=1,
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(settings, delay_tracker=tracker, webhook_notifier=notifier),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/reading.json"
        early = json.loads(urlopen(url).read())
        assert early["issue"] == "low_pressure"
        assert not early["delay"]["eligible"]
        assert not requests

        current_time[0] = 10.0
        due = json.loads(urlopen(url).read())
        retry = json.loads(urlopen(url).read())
        repeated = json.loads(urlopen(url).read())

        assert due["delay"]["eligible"]
        assert due["delay"]["notification_due"]
        assert retry["delay"]["notification_due"]
        assert not repeated["delay"]["notification_due"]
        assert repeated["delay"]["eligible"]
        assert len(requests) == 2
        assert requests[0]["status"] == "low"
        assert requests[0]["pressure"] == 1.3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_manual_capture_endpoint_grabs_frame_and_returns_timestamp(tmp_path) -> None:
    frame = cv2.imread(str(Path(__file__).with_name("boiler.PNG")))
    image_path = tmp_path / "latest.jpg"
    poller = CameraPoller(
        rtsp_url="rtsp://camera.invalid/stream",
        username="",
        password="",
        image_path=image_path,
        poll_seconds=30,
        capture_fn=lambda: frame,
        logger=lambda event, **fields: None,
    )
    settings = ViewerSettings(image_path=image_path, roi=ROI(330, 238, 110, 94))
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(settings, camera_poller=poller),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/capture",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urlopen(request)
        result = json.loads(response.read())

        assert response.status == 200
        assert result["captured"]
        assert result["camera"]["available"]
        assert result["camera"]["checked_at"]
        assert image_path.is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_manual_test_webhook_endpoint_sends_simulated_error(tmp_path) -> None:
    requests = []
    notifier = HomeAssistantWebhookNotifier(
        "http://ha.local/api/webhook/test-id",
        post_json=lambda url, payload, timeout: requests.append(payload) or 200,
    )
    settings = ViewerSettings(
        image_path=tmp_path / "latest.jpg",
        roi=ROI(0, 0, 50, 50),
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(settings, webhook_notifier=notifier),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/test-webhook",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urlopen(request)
        result = json.loads(response.read())

        assert response.status == 200
        assert result["delivered"]
        assert requests == [{
            "status": "invalid",
            "issue": "error",
            "pressure": None,
            "display": "ECS",
            "error_code": "ECS",
            "error": "Manual test message",
            "delay_minutes": 0,
            "simulated": True,
        }]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()