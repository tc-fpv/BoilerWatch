"""HTTP layer: routes each request to the state, analysis and service objects."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Type
from urllib.parse import urlsplit

from boiler_reader.activity import GRABBING, IDLE
from boiler_reader.analysis import FrameAnalyzer, camera_status
from boiler_reader.frames import encode_jpeg, load_frame, to_bgr
from boiler_reader.roi_service import RoiService
from boiler_reader.screenshots import save_mismatch
from boiler_reader.settings import SETTINGS_KEYS
from boiler_reader.settings_service import SettingsService
from boiler_reader.state import AppState

_STATIC_DIR = Path(__file__).with_name("static")
_STATIC_IMAGES = {
    "/logo.png": "logo.png",
    "/logo-text.png": "logo-text.png",
    "/favicon.png": "favicon.png",
}
_STATIC_TEXT = {
    "/page.css": ("page.css", "text/css; charset=utf-8"),
    "/page.js": ("page.js", "text/javascript; charset=utf-8"),
}

# path -> handler method
_GET_ROUTES = {
    "/": "_get_page",
    "/favicon.ico": "_get_favicon",
    "/reading.json": "_get_reading",
    "/settings.json": "_get_settings",
    "/admin-settings": "_get_admin_settings",
    "/activity.json": "_get_activity",
    "/camera-status.json": "_get_camera_status",
    "/llm-image.jpg": "_get_llm_image",
    "/last.jpg": "_get_last_image",
}

# path -> (handler method, what a failed save is called, or None to leave errors to the handler)
_POST_ROUTES = {
    "/capture": ("_post_capture", None),
    "/test-webhook": ("_post_test_webhook", None),
    "/save-mismatch": ("_post_save_mismatch", "image"),
    "/settings": ("_post_settings", "settings"),
    "/admin-settings": ("_post_admin_settings", "admin settings"),
    "/roi": ("_post_roi", "ROI"),
}


@lru_cache(maxsize=1)
def _build_commit() -> str:
    """Short commit id: BUILD_COMMIT (set at image build), else local git, else 'dev'."""
    baked = os.getenv("BUILD_COMMIT", "").strip()
    if baked and baked != "unknown":
        return baked[:7]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return result.stdout.strip() or "dev"
    except (OSError, subprocess.SubprocessError):
        return "dev"


def _page(refresh_seconds: int) -> bytes:
    html = (_STATIC_DIR / "page.html").read_text(encoding="utf-8")
    return (
        html.replace("__REFRESH_MS__", str(refresh_seconds * 1000))
        .replace("__BUILD_LABEL__", f"Build {_build_commit()}")
        .encode("utf-8")
    )


class _Handler(BaseHTTPRequestHandler):
    # Bound to real objects by make_handler.
    state: AppState
    analyzer: FrameAnalyzer
    settings_service: SettingsService
    roi_service: RoiService

    # -- dispatch ---------------------------------------------------------

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in _STATIC_IMAGES:
            self._send_static(_STATIC_IMAGES[path], "image/png", "public, max-age=86400")
        elif path in _STATIC_TEXT:
            filename, content_type = _STATIC_TEXT[path]
            self._send_static(filename, content_type, "no-cache")
        elif path in _GET_ROUTES:
            getattr(self, _GET_ROUTES[path])()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in _POST_ROUTES:
            self.send_error(404)
            return
        method, save_label = _POST_ROUTES[path]
        if save_label is None:
            getattr(self, method)()
            return
        try:
            getattr(self, method)()
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except OSError as error:
            self._send_json(500, {"error": f"Could not save {save_label}: {error}"})

    # -- GET --------------------------------------------------------------

    def _get_favicon(self) -> None:
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _get_page(self) -> None:
        self._send_bytes(
            _page(self.state.settings.refresh_seconds), "text/html; charset=utf-8"
        )

    def _get_reading(self) -> None:
        state, settings = self.state, self.state.settings
        cached = state.get_cached_reading()
        if cached is not None:
            # Re-evaluate the delay tracker and webhook on each poll
            # so the issue timer advances and alerts fire on schedule.
            self._send_json(200, self.analyzer.refresh_cached(cached))
            return
        # A grab or analysis is already under way and will fill the
        # cache; decoding the old frame here would send a duplicate
        # request to the LLM.
        if state.activity.snapshot()["phase"] != IDLE:
            self._send_json(200, {"busy": True})
            return
        # No cache yet — decode once on first request.
        image = load_frame(settings.image_path)
        if image is None:
            camera = camera_status(settings.image_path)
            delay = self.analyzer.fail("stale", str(camera.get("error") or "Camera frame unavailable"))
            self._send_json(
                503,
                {
                    "error": camera.get("error") or "Latest image is not available",
                    "camera": camera,
                    "issue": "error",
                    "delay": asdict(delay),
                },
            )
            return
        try:
            with state.decode_lock:
                # Concurrent requests share one decode.
                reading = state.get_cached_reading() or self.analyzer.analyze_on_request(image)
        except (ValueError, RuntimeError) as error:
            self._send_json(500, {"error": str(error)})
            return
        self._send_json(200, reading)

    def _get_settings(self) -> None:
        self._send_json(200, self.state.snapshot_settings())

    def _get_admin_settings(self) -> None:
        settings = self.state.snapshot_settings()
        self._send_json(200, {k: settings[k] for k in SETTINGS_KEYS if k in settings})

    def _get_activity(self) -> None:
        self._send_json(200, self.state.activity.snapshot())

    def _get_camera_status(self) -> None:
        self._send_json(200, camera_status(self.state.settings.image_path))

    def _get_llm_image(self) -> None:
        body = self.state.llm_image
        if body is None:
            self.send_error(404, "No LLM image yet")
            return
        self._send_bytes(body, "image/jpeg")

    def _get_last_image(self) -> None:
        image = load_frame(self.state.settings.image_path)
        if image is None:
            self.send_error(503, "Camera has not supplied a frame")
            return
        try:
            # The ROI box is drawn by the page, so serve the frame unannotated.
            body = encode_jpeg(to_bgr(image))
        except ValueError as error:
            self.send_error(500, str(error))
            return
        self._send_bytes(body, "image/jpeg")

    # -- POST -------------------------------------------------------------

    def _post_capture(self) -> None:
        state = self.state
        if state.camera_poller is None:
            self._send_json(503, {"error": "RTSP_URL is not configured"})
            return
        # Mark the grab as started before clearing the cache so a
        # concurrent refresh doesn't decode the old frame.
        state.activity.set(GRABBING)
        # Clear stale cache before capture so the fresh frame is decoded.
        state.set_cached_reading(None)
        captured = state.camera_poller.capture_once()
        camera = camera_status(state.settings.image_path)
        self._send_json(
            200 if captured else 502,
            {"captured": captured, "camera": camera, "error": camera.get("error")},
        )

    def _post_test_webhook(self) -> None:
        notifier = self.state.notifier
        if notifier is None:
            self._send_json(503, {"error": "HA_WEBHOOK_URL is not configured"})
            return
        delivered = notifier.send(
            {
                "status": "invalid",
                "issue": "error",
                "pressure": None,
                "display": "ECS",
                "error_code": "ECS",
                "error": "Manual test message",
                "delay_minutes": 0,
                "simulated": True,
            }
        )
        self._send_json(
            200 if delivered else 502,
            {"delivered": delivered, "error": None if delivered else "Webhook delivery failed"},
        )

    def _post_save_mismatch(self) -> None:
        payload = self._read_json(512, "Request body")
        filename = save_mismatch(
            self.state.settings, self.state.get_cached_reading(), payload
        )
        if filename is None:
            self._send_json(503, {"error": "No current image to save"})
            return
        self._send_json(200, {"saved": filename})

    def _post_settings(self) -> None:
        payload = self._read_json(1024, "Settings request body")
        self._send_json(200, self.settings_service.update_reader(payload))

    def _post_admin_settings(self) -> None:
        payload = self._read_json(4096, "Admin settings body")
        self.settings_service.update_admin(payload)
        self._send_json(200, {"ok": True})

    def _post_roi(self) -> None:
        payload = self._read_json(1024, "ROI request body")
        result = self.roi_service.set_roi(payload)
        if result is None:
            self._send_json(503, {"error": "Latest image is not available"})
            return
        self._send_json(200, result)

    # -- helpers ----------------------------------------------------------

    def _read_json(self, max_bytes: int, name: str) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if not 0 < content_length <= max_bytes:
            raise ValueError(f"{name} must be between 1 and {max_bytes} bytes")
        return json.loads(self.rfile.read(content_length))

    def _send_static(self, filename: str, content_type: str, cache_control: str) -> None:
        self._send_bytes((_STATIC_DIR / filename).read_bytes(), content_type, cache_control)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        cache_control: str = "no-store",
        status: int = 200,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def log_message(self, format: str, *args: object) -> None:
        pass


def make_handler(
    state: AppState,
    analyzer: FrameAnalyzer | None = None,
    settings_service: SettingsService | None = None,
) -> Type[BaseHTTPRequestHandler]:
    """Build the request handler for an AppState, with default services if none given."""
    analyzer = analyzer or FrameAnalyzer(state)
    return type(
        "ImageHandler",
        (_Handler,),
        {
            "state": state,
            "analyzer": analyzer,
            "settings_service": settings_service or SettingsService(state),
            "roi_service": RoiService(state, analyzer),
        },
    )
