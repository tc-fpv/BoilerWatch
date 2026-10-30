"""Run the BoilerWatch server: wire the camera, analysis and web layers together."""

from __future__ import annotations

import logging
import uvicorn


from boiler_reader.activity import GRABBING, IDLE
from boiler_reader.analysis import FrameAnalyzer
from boiler_reader.camera_service import CameraService
from boiler_reader.settings import ViewerSettings, settings_from_env
from boiler_reader.settings_service import SettingsService, make_notifier
from boiler_reader.state import AppState
from boiler_reader.web import create_app


def serve(settings: ViewerSettings | None = None) -> None:
    settings = settings or settings_from_env()
    state = AppState(
        settings,
        notifier=make_notifier({
            "ha_webhook_url": settings.ha_webhook_url,
            "ha_webhook_timeout": settings.ha_webhook_timeout,
        }),
    )
    logging.getLogger().addHandler(state.log_buffer)
    analyzer = FrameAnalyzer(state)
    camera = CameraService(
        settings.image_path,
        on_frame=analyzer.analyze_captured,
        on_grab_state=lambda grabbing: state.activity.set(GRABBING if grabbing else IDLE),
        on_poller=lambda poller: setattr(state, "camera_poller", poller),
    )
    settings_service = SettingsService(state, restart_camera=camera.start)

    if settings.rtsp_url:
        camera.start(
            settings.rtsp_url,
            settings.rtsp_user,
            settings.rtsp_pass,
            settings.poll_seconds,
            first_capture_in_background=True,
        )

    print(f"boilerwatch: serving on port {settings.port}; image={settings.image_path}")
    try:
        uvicorn.run(
            create_app(state, analyzer, settings_service),
            host="0.0.0.0",
            port=settings.port,
            log_config=None,
            access_log=False,
        )
    finally:
        camera.stop()


def main() -> None:
    # The modules log ready-made JSON lines; without a handler INFO is dropped.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    serve()


if __name__ == "__main__":
    main()
