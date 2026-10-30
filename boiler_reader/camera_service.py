"""Owns the camera polling thread and restarts it when the camera settings change."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable

from boiler_reader.camera import CameraPoller, FrameCallback


class CameraService:
    def __init__(
        self,
        image_path: Path,
        *,
        on_frame: FrameCallback,
        on_grab_state: Callable[[bool], None],
        on_poller: Callable[[CameraPoller], None] | None = None,
        poller_factory: Callable[..., CameraPoller] = CameraPoller,
    ) -> None:
        self._image_path = image_path
        self._on_frame = on_frame
        self._on_grab_state = on_grab_state
        self._on_poller = on_poller
        self._poller_factory = poller_factory
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(
        self,
        rtsp_url: str,
        rtsp_user: str,
        rtsp_pass: str,
        poll_seconds: int,
        first_capture_in_background: bool = False,
    ) -> None:
        """(Re)start polling.  In background mode the first grab runs in the thread."""
        with self._lock:
            self._stop_locked()
            self._stop_event = Event()
            poller = self._poller_factory(
                rtsp_url=rtsp_url,
                username=rtsp_user,
                password=rtsp_pass,
                image_path=self._image_path,
                poll_seconds=poll_seconds,
                on_frame=self._on_frame,
                on_grab_state=self._on_grab_state,
            )
            if self._on_poller is not None:
                self._on_poller(poller)
            if first_capture_in_background:
                # Let the page load straight away; it shows "Initialising"
                # while this first grab and analysis run.
                self._on_grab_state(True)
            else:
                poller.capture_once()
            self._thread = Thread(
                target=poller.run,
                kwargs={
                    "stop_event": self._stop_event,
                    "capture_on_start": first_capture_in_background,
                },
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
