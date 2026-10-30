"""Loading, converting, encoding and annotating camera frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from boiler_reader.decoder import ROI


def load_frame(path: Path) -> np.ndarray | None:
    """The stored frame exactly as saved, or None if it can't be read."""
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """A BGR copy of a grayscale, BGR or BGRA image."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError("image must be grayscale, BGR, or BGRA")


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode latest image")
    return encoded.tobytes()


def annotate_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    """Return a copy of an image with the ROI boundary highlighted."""
    height, width = image.shape[:2]
    if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI must have a non-negative origin and positive dimensions")
    if roi.x + roi.width > width or roi.y + roi.height > height:
        raise ValueError("ROI extends beyond the image")

    annotated = to_bgr(image)
    thickness = max(1, round(min(width, height) / 300))
    cv2.rectangle(
        annotated,
        (roi.x, roi.y),
        (roi.x + roi.width - 1, roi.y + roi.height - 1),
        (0, 100, 255),
        thickness,
    )
    return annotated
