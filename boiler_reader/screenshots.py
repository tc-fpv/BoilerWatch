"""Saving a frame, with the user's correction, when the reading was wrong."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import cv2

from boiler_reader.frames import load_frame
from boiler_reader.locator import crop_display, locate_display
from boiler_reader.settings import ViewerSettings


def _safe(text: str, limit: int) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "", text)[:limit]


def save_mismatch(
    settings: ViewerSettings,
    cached: dict[str, object] | None,
    payload: dict[str, object],
) -> str | None:
    """Save the current frame (and the crop the LLM saw) named after the reading and the
    user's correction.  Returns the filename, or None if there is no current frame.

    Raises ValueError for a bad payload and OSError if the files can't be written.
    """
    value = str(payload.get("value", "")).strip()
    unit = str(payload.get("unit", "")).strip()
    if not value:
        raise ValueError("value is required")
    if unit not in ("bar", "degC", "error"):
        raise ValueError("unit must be bar, degC, or error")
    image = load_frame(settings.image_path)
    if image is None:
        return None

    if cached is not None:
        llm_seg = (
            f"{_safe(str(cached.get('display_text') or 'unknown'), 20)}"
            f"_{_safe(str(cached.get('unit') or 'unknown'), 10)}"
        )
    else:
        llm_seg = "unknown"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = settings.image_path.parent / "screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{stamp}_LLM_{llm_seg}_MANUAL_{_safe(value, 20)}_{unit}.jpg"
    cv2.imwrite(str(folder / filename), image)
    # Also keep the crop the LLM was shown, to diagnose misreads.
    located = locate_display(image, settings.brightness_threshold)
    if located is not None:
        cv2.imwrite(
            str(folder / filename.replace(".jpg", "_ocr.jpg")),
            crop_display(image, located),
        )
    return filename
