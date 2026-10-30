"""Direct RTSP camera polling and latest-frame status management."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from typing import Callable
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np


CaptureFunction = Callable[[], np.ndarray | None]
FrameCallback = Callable[[np.ndarray | None, str | None], None]


class CameraPoller:
    def __init__(
        self,
        rtsp_url: str,
        username: str,
        password: str,
        image_path: Path,
        poll_seconds: int,
        capture_fn: CaptureFunction | None = None,
        logger: Callable[..., None] | None = None,
        on_frame: FrameCallback | None = None,
        on_grab_state: Callable[[bool], None] | None = None,
    ) -> None:
        self.on_grab_state = on_grab_state
        self.rtsp_url = rtsp_url
        self.username = username
        self.password = password
        self.image_path = Path(image_path)
        self.status_path = self.image_path.with_name("camera-status.json")
        self.poll_seconds = poll_seconds
        self._poll_lock = Lock()
        self._capture_lock = Lock()
        self._poll_wakeup = Event()
        self.capture_fn = capture_fn or self._capture_rtsp
        self.logger = logger or self._log
        self.on_frame = on_frame

    @staticmethod
    def _log(event: str, **fields: object) -> None:
        logging.getLogger("boiler_reader.camera").info(
            json.dumps({"event": event, **fields}, separators=(",", ":"))
        )

    def _authenticated_url(self) -> str:
        parts = urlsplit(self.rtsp_url)
        if not self.username or "@" in parts.netloc:
            return self.rtsp_url
        credentials = quote(self.username, safe="")
        if self.password:
            credentials += ":" + quote(self.password, safe="")
        return urlunsplit(
            (parts.scheme, f"{credentials}@{parts.netloc}", parts.path, parts.query, parts.fragment)
        )

    def _capture_rtsp(self) -> np.ndarray:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        capture = cv2.VideoCapture()
        opened = capture.open(
            self._authenticated_url(),
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                10000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                10000,
            ],
        )
        if not opened:
            capture.release()
            raise RuntimeError("Could not open the configured RTSP camera stream")
        try:
            received, frame = capture.read()
        finally:
            capture.release()
        if not received or frame is None or frame.size == 0:
            raise RuntimeError("RTSP camera did not return a frame")
        return frame

    def _redact(self, message: str) -> str:
        """Hide credentials; a very short username is skipped as it would mangle the text."""
        secrets = [self.rtsp_url, self.password]
        if len(self.username) >= 4:
            secrets.append(self.username)
        for secret in secrets:
            if secret:
                message = message.replace(secret, "<REDACTED>")
        return message

    def _write_status(self, available: bool, error: str | None = None) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        status = {
            "available": available,
            "error": error,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        temporary_path = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(status), encoding="utf-8")
        temporary_path.replace(self.status_path)

    def _clear_latest_frame(self) -> None:
        try:
            self.image_path.unlink(missing_ok=True)
        except OSError as error:
            self.logger("camera_image_remove_failed", error=str(error))

    def capture_once(self) -> bool:
        with self._capture_lock:
            # Held for the whole capture (grab, save and on_frame analysis) so
            # observers never see an idle gap between the stages.
            if self.on_grab_state is not None:
                self.on_grab_state(True)
            try:
                return self._capture_once()
            finally:
                if self.on_grab_state is not None:
                    self.on_grab_state(False)

    def _grab_and_store(self) -> np.ndarray:
        """Grab a frame and save it as the latest image; raises if either step fails."""
        frame = self.capture_fn()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise RuntimeError("RTSP camera did not return a frame")
        extension = self.image_path.suffix.lower() or ".jpg"
        encoded_ok, encoded = cv2.imencode(extension, frame)
        if not encoded_ok:
            raise RuntimeError("Could not encode the camera frame")
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.image_path.with_suffix(self.image_path.suffix + ".tmp")
        temporary_path.write_bytes(encoded.tobytes())
        temporary_path.replace(self.image_path)
        self._write_status(available=True)
        return frame

    def recapture(self) -> np.ndarray | None:
        """Grab a replacement frame from inside an on_frame callback (the capture lock is
        already held).  Returns None, leaving the previous frame in place, if the grab fails."""
        try:
            return self._grab_and_store()
        except Exception as error:
            self.logger("camera_regrab_failed", error=self._redact(str(error)))
            return None

    def _capture_once(self) -> bool:
        try:
            frame = self._grab_and_store()
        except Exception as error:
            self._clear_latest_frame()
            message = self._redact(str(error))
            try:
                self._write_status(available=False, error=message)
            except OSError as status_error:
                self.logger("camera_status_write_failed", error=str(status_error))
            if self.on_frame is not None:
                try:
                    self.on_frame(None, message)
                except Exception as callback_error:
                    self.logger("camera_frame_callback_failed", error=str(callback_error))
            self.logger("camera_frame_failed", error=message)
            return False

        # Outside the try above so a failing callback or logger can't discard a good frame.
        if self.on_frame is not None:
            try:
                self.on_frame(frame, None)
            except Exception as callback_error:
                self.logger("camera_frame_callback_failed", error=str(callback_error))
        self.logger(
            "camera_frame_captured",
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
        )
        return True

    def run(self, stop_event: Event, capture_on_start: bool = True) -> None:
        self.run_with_wait(stop_event, capture_on_start, None)

    def run_with_wait(
        self,
        stop_event: Event,
        capture_on_start: bool,
        wait_fn: Callable[[float], bool] | None,
    ) -> None:
        cycle_started = time.monotonic()
        if capture_on_start:
            self.capture_once()
        wait = wait_fn or self._poll_wakeup.wait
        while not stop_event.is_set():
            # Poll start-to-start, so analysis time doesn't stretch the interval.
            remaining = max(0.0, self.get_poll_seconds() - (time.monotonic() - cycle_started))
            woke_early = wait(remaining)
            if stop_event.is_set():
                break
            if woke_early:
                self._poll_wakeup.clear()
                continue
            cycle_started = time.monotonic()
            self.capture_once()

    def wake(self) -> None:
        self._poll_wakeup.set()

    def get_poll_seconds(self) -> int:
        with self._poll_lock:
            return self.poll_seconds

    def set_poll_seconds(self, poll_seconds: int) -> None:
        if not 1 <= poll_seconds <= 3600:
            raise ValueError("POLL_SECONDS must be between 1 and 3600")
        with self._poll_lock:
            self.poll_seconds = poll_seconds
        self._poll_wakeup.set()