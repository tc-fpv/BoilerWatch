"""HTTP layer (FastAPI): routes each request to the state, analysis and service objects."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

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


class _BadRequest(Exception):
    pass


def _json_body(max_bytes: int, name: str) -> Callable[[Request], object]:
    """Dependency: the request's JSON object, rejecting empty or oversized bodies."""

    async def read(request: Request) -> dict[str, object]:
        raw = await request.body()
        if not 0 < len(raw) <= max_bytes:
            raise _BadRequest(f"{name} must be between 1 and {max_bytes} bytes")
        try:
            return json.loads(raw)
        except ValueError as error:
            raise _BadRequest(str(error)) from error

    return read


def _json(status: int, payload: dict[str, object]) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})


def _bytes(body: bytes, content_type: str, cache_control: str = "no-store") -> Response:
    return Response(body, media_type=content_type, headers={"Cache-Control": cache_control})


def _static(filename: str, content_type: str, cache_control: str) -> Response:
    return _bytes((_STATIC_DIR / filename).read_bytes(), content_type, cache_control)


def _saving(label: str, action: Callable[[], Response]) -> Response:
    """Run a settings/ROI/image save, turning bad input into 400 and write failures into 500."""
    try:
        return action()
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return _json(400, {"error": str(error)})
    except OSError as error:
        return _json(500, {"error": f"Could not save {label}: {error}"})


def create_app(
    state: AppState,
    analyzer: FrameAnalyzer | None = None,
    settings_service: SettingsService | None = None,
) -> FastAPI:
    """Build the web app for an AppState, with default services if none given."""
    analyzer = analyzer or FrameAnalyzer(state)
    settings_service = settings_service or SettingsService(state)
    roi_service = RoiService(state, analyzer)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(_BadRequest)
    def bad_request(request: Request, error: _BadRequest) -> JSONResponse:
        return _json(400, {"error": str(error)})

    # -- GET --------------------------------------------------------------

    @app.get("/")
    def page() -> Response:
        return _bytes(_page(state.settings.refresh_seconds), "text/html; charset=utf-8")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    for path, filename in _STATIC_IMAGES.items():
        app.add_api_route(
            path,
            lambda filename=filename: _static(filename, "image/png", "public, max-age=86400"),
            methods=["GET"],
        )
    for path, (filename, content_type) in _STATIC_TEXT.items():
        app.add_api_route(
            path,
            lambda filename=filename, content_type=content_type: _static(
                filename, content_type, "no-cache"
            ),
            methods=["GET"],
        )

    @app.get("/reading.json")
    def reading() -> Response:
        settings = state.settings
        cached = state.get_cached_reading()
        if cached is not None:
            # Re-evaluate the delay tracker and webhook on each poll
            # so the issue timer advances and alerts fire on schedule.
            return _json(200, analyzer.refresh_cached(cached))
        # A grab or analysis is already under way and will fill the
        # cache; decoding the old frame here would send a duplicate
        # request to the LLM.
        if state.activity.snapshot()["phase"] != IDLE:
            return _json(200, {"busy": True})
        # No cache yet — decode once on first request.
        image = load_frame(settings.image_path)
        if image is None:
            camera = camera_status(settings.image_path)
            delay = analyzer.fail("stale", str(camera.get("error") or "Camera frame unavailable"))
            return _json(
                503,
                {
                    "error": camera.get("error") or "Latest image is not available",
                    "camera": camera,
                    "issue": "error",
                    "delay": asdict(delay),
                },
            )
        try:
            with state.decode_lock:
                # Concurrent requests share one decode.
                result = state.get_cached_reading() or analyzer.analyze_on_request(image)
        except (ValueError, RuntimeError) as error:
            return _json(500, {"error": str(error)})
        return _json(200, result)

    @app.get("/settings.json")
    def settings_json() -> Response:
        return _json(200, state.snapshot_settings())

    @app.get("/admin-settings")
    def admin_settings() -> Response:
        settings = state.snapshot_settings()
        return _json(200, {k: settings[k] for k in SETTINGS_KEYS if k in settings})

    @app.get("/activity.json")
    def activity() -> Response:
        return _json(200, state.activity.snapshot())

    @app.get("/camera-status.json")
    def camera_status_json() -> Response:
        return _json(200, camera_status(state.settings.image_path))

    @app.get("/llm-image.jpg")
    def llm_image() -> Response:
        body = state.llm_image
        if body is None:
            return Response("No LLM image yet", status_code=404)
        return _bytes(body, "image/jpeg")

    @app.get("/last.jpg")
    def last_image() -> Response:
        image = load_frame(state.settings.image_path)
        if image is None:
            return Response("Camera has not supplied a frame", status_code=503)
        try:
            # The ROI box is drawn by the page, so serve the frame unannotated.
            body = encode_jpeg(to_bgr(image))
        except ValueError as error:
            return Response(str(error), status_code=500)
        return _bytes(body, "image/jpeg")

    # -- POST -------------------------------------------------------------

    @app.post("/capture")
    def capture() -> Response:
        if state.camera_poller is None:
            return _json(503, {"error": "RTSP_URL is not configured"})
        # Mark the grab as started before clearing the cache so a
        # concurrent refresh doesn't decode the old frame.
        state.activity.set(GRABBING)
        # Clear stale cache before capture so the fresh frame is decoded.
        state.set_cached_reading(None)
        captured = state.camera_poller.capture_once()
        camera = camera_status(state.settings.image_path)
        return _json(
            200 if captured else 502,
            {"captured": captured, "camera": camera, "error": camera.get("error")},
        )

    @app.post("/test-webhook")
    def test_webhook() -> Response:
        notifier = state.notifier
        if notifier is None:
            return _json(503, {"error": "HA_WEBHOOK_URL is not configured"})
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
        return _json(
            200 if delivered else 502,
            {"delivered": delivered, "error": None if delivered else "Webhook delivery failed"},
        )

    @app.post("/save-mismatch")
    def save_mismatch_route(payload: dict = Depends(_json_body(512, "Request body"))) -> Response:
        def save() -> Response:
            filename = save_mismatch(state.settings, state.get_cached_reading(), payload)
            if filename is None:
                return _json(503, {"error": "No current image to save"})
            return _json(200, {"saved": filename})

        return _saving("image", save)

    @app.post("/settings")
    def settings_route(payload: dict = Depends(_json_body(1024, "Settings request body"))) -> Response:
        return _saving("settings", lambda: _json(200, settings_service.update_reader(payload)))

    @app.post("/admin-settings")
    def admin_settings_route(payload: dict = Depends(_json_body(4096, "Admin settings body"))) -> Response:
        def save() -> Response:
            settings_service.update_admin(payload)
            return _json(200, {"ok": True})

        return _saving("admin settings", save)

    @app.post("/roi")
    def roi_route(payload: dict = Depends(_json_body(1024, "ROI request body"))) -> Response:
        def save() -> Response:
            result = roi_service.set_roi(payload)
            if result is None:
                return _json(503, {"error": "Latest image is not available"})
            return _json(200, result)

        return _saving("ROI", save)

    return app
