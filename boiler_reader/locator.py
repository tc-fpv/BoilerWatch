"""Automatic display localisation for dark-background camera frames.

Finds the self-illuminated seven-segment display by brightness thresholding,
separates the main digit row from indicator icons, corrects for modest tilt,
and returns digit/decimal box coordinates ready for the segment decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from boiler_reader.decoder import DecoderConfig, DigitBox, ROI


# Minimum fraction of frame width that a bright region must span to be
# considered the OSD timestamp strip rather than part of the gauge display.
_OSD_MIN_WIDTH_FRACTION = 0.40

# Bright regions whose bottom edge falls within this fraction of the frame
# height from the bottom are candidates for the OSD timestamp strip.
_OSD_BOTTOM_FRACTION = 0.20

# Pixels brighter than this (0-255) are considered "lit" by the display.
DEFAULT_BRIGHTNESS_THRESHOLD = 30

# Minimum number of bright pixels for a contour to be considered real signal.
# Set high enough to exclude isolated camera artefacts (IR LEDs, reflections).
_MIN_BRIGHT_AREA = 500

# The digit row is the topmost tall region; indicator icons sit below it.
# Regions whose height is less than this fraction of the tallest region are
# treated as sub-display indicators and excluded.
_MIN_HEIGHT_FRACTION = 0.45

# How much padding (as a fraction of the detected digit height) to add around
# each inferred digit box to absorb blur / slight misalignment.
_BOX_PADDING_FRACTION = 0.08


@dataclass(frozen=True)
class LocatedDisplay:
    """Result of automatic display localisation."""
    roi: ROI
    digit_boxes: tuple[DigitBox, ...]
    decimal_box: DigitBox | None
    confidence: float


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _brightness_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _is_osd_strip(contour: np.ndarray, frame_width: int, frame_height: int) -> bool:
    """Return True if a contour looks like the camera OSD timestamp bar."""
    x, y, w, h = cv2.boundingRect(contour)
    wide = w >= frame_width * _OSD_MIN_WIDTH_FRACTION
    near_bottom = (y + h) >= frame_height * (1.0 - _OSD_BOTTOM_FRACTION)
    return wide and near_bottom


def _find_digit_row_contours(
    mask: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> list[np.ndarray]:
    """Return contours that belong to the main digit display row."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [
        c for c in contours
        if cv2.contourArea(c) >= _MIN_BRIGHT_AREA
        and not _is_osd_strip(c, frame_width, frame_height)
    ]
    if not candidates:
        return []

    # Keep only the tallest cluster; drop indicator icons below the digit row.
    # Height alone is not enough: unit indicators can be nearly as tall as a
    # digit, so also require the blob's vertical centre to sit inside the
    # tallest blob's vertical extent.
    tallest = max(candidates, key=lambda c: cv2.boundingRect(c)[3])
    _, row_y, _, max_h = cv2.boundingRect(tallest)
    digit_row = []
    for c in candidates:
        _, y, _, h = cv2.boundingRect(c)
        centre_y = y + h / 2
        if h >= max_h * _MIN_HEIGHT_FRACTION and row_y <= centre_y <= row_y + max_h:
            digit_row.append(c)
    return digit_row


def _merge_bounding_rect(contours: list[np.ndarray]) -> tuple[int, int, int, int]:
    """Return the bounding rect that encompasses all supplied contours."""
    rects = [cv2.boundingRect(c) for c in contours]
    x = min(rx for rx, _, _, _ in rects)
    y = min(ry for _, ry, _, _ in rects)
    x2 = max(rx + rw for rx, _, rw, _ in rects)
    y2 = max(ry + rh for _, ry, _, rh in rects)
    return x, y, x2 - x, y2 - y


def _lit_runs(col_sums: np.ndarray, min_width: int = 4) -> list[tuple[int, int]]:
    """Return (start, end) of contiguous lit column runs."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(col_sums):
        if v > 0 and not in_run:
            in_run = True
            start = i
        elif v == 0 and in_run:
            in_run = False
            if i - start >= min_width:
                runs.append((start, i))
    if in_run and len(col_sums) - start >= min_width:
        runs.append((start, len(col_sums)))
    return runs


def _infer_digit_boxes(
    crop: np.ndarray,
    threshold: int,
) -> tuple[tuple[DigitBox, ...], DigitBox | None]:
    """Infer per-character boxes and decimal box from a digit-row crop.

    Count lit column runs to find how many characters are present (2 or 3).
    Use the widest run as the reference cell width, then place all cells
    right-aligned to the rightmost run end with equal spacing.  This handles
    narrow glyphs like '1' whose lit pixels are much narrower than their cell.
    """
    _, binary = cv2.threshold(crop, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    h, w = binary.shape
    min_char_width = max(4, int(w * 0.06))

    col_sums = binary.sum(axis=0).astype(np.int64)
    runs = _lit_runs(col_sums, min_width=min_char_width)
    n_chars = len(runs)

    if n_chars not in (2, 3):
        # Fallback: split lit extent into equal thirds.
        lit_cols = [i for i, v in enumerate(col_sums) if v > 0]
        span_start = lit_cols[0] if lit_cols else 0
        span_end = lit_cols[-1] + 1 if lit_cols else w
        n_chars = 3
        cell_w = (span_end - span_start) // n_chars
        runs = [
            (span_start + i * cell_w, span_start + (i + 1) * cell_w)
            for i in range(n_chars)
        ]

    # Use the widest run's width as the reference cell size, then expand every
    # run to that width, anchoring each to its own centre.  This means a narrow
    # '1' glyph gets a box the same width as the adjacent '2', keeping segment
    # zone proportions valid across all glyphs.
    cell_w = max(e - s for s, e in runs)
    boundaries = [
        (s + (e - s) // 2 - cell_w // 2, s + (e - s) // 2 - cell_w // 2 + cell_w)
        for s, e in runs
    ]

    digit_boxes = tuple(
        DigitBox(
            x=max(0, s),
            y=0,
            width=min(w, e) - max(0, s),
            height=h,
        )
        for s, e in boundaries
    )

    # Decimal box: bottom strip of the last character cell.
    last_s, last_e = boundaries[-1]
    dp_h = max(4, h // 4)
    decimal_box = DigitBox(
        x=max(0, last_s),
        y=max(0, h - dp_h),
        width=min(w, last_e) - max(0, last_s),
        height=dp_h,
    )

    return digit_boxes, decimal_box


def _refine_digit_row(
    gray: np.ndarray,
    rect: tuple[int, int, int, int],
    threshold: int,
) -> tuple[int, int, int, int] | None:
    """Re-find the digit row inside ``rect`` at a stricter, local threshold.

    On overexposed frames the glow around a digit can bridge to a lit unit
    indicator below it at the global threshold, merging both into one blob.
    Otsu on the candidate region separates the saturated segments from the
    glow, so the digit row can be found again without the indicator.  Returns
    the refined rect in frame coordinates, or None to keep the original.  The
    caller uses only its vertical extent.
    """
    x, y, w, h = rect
    crop = gray[y:y + h, x:x + w]
    if crop.size == 0 or int(crop.min()) == int(crop.max()):
        return None
    otsu, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if int(otsu) <= threshold:
        return None

    mask = _brightness_mask(crop, int(otsu))
    # Treat the crop as its own frame; the OSD strip can't be inside it.
    contours = _find_digit_row_contours(mask, crop.shape[1] * 100, crop.shape[0])
    if not contours:
        return None
    rx, ry, rw, rh = _merge_bounding_rect(contours)
    return x + rx, y + ry, rw, rh


def _confidence(mask: np.ndarray, roi_x: int, roi_y: int, roi_w: int, roi_h: int) -> float:
    """Fraction of bright pixels in the frame that fall inside the detected ROI."""
    total_bright = int(np.count_nonzero(mask))
    if total_bright == 0:
        return 0.0
    roi_bright = int(np.count_nonzero(
        mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    ))
    return roi_bright / total_bright


def locate_display(
    image: np.ndarray,
    brightness_threshold: int = DEFAULT_BRIGHTNESS_THRESHOLD,
) -> LocatedDisplay | None:
    """Find the seven-segment display in a dark-background camera frame.

    Returns a LocatedDisplay with ROI, digit boxes, and decimal box, or None
    if no display could be found.
    """
    frame_h, frame_w = image.shape[:2]
    gray = _to_grayscale(image)
    mask = _brightness_mask(gray, brightness_threshold)

    contours = _find_digit_row_contours(mask, frame_w, frame_h)
    if not contours:
        return None

    bx, by, bw, bh = _merge_bounding_rect(contours)
    refined = _refine_digit_row(gray, (bx, by, bw, bh), brightness_threshold)
    if refined is not None:
        # Only trim vertically: the bridged indicator sits below the digits, and
        # the stricter threshold can drop a dim, thin trailing "1" horizontally.
        _, by, _, bh = refined
    pad_px = max(2, int(bh * _BOX_PADDING_FRACTION))
    # Expand the bounding rect by padding in all directions.
    roi_x = max(0, bx - pad_px)
    roi_y = max(0, by - pad_px)
    roi_x2 = min(frame_w, bx + bw + pad_px)
    roi_y2 = min(frame_h, by + bh + pad_px)
    roi_w = roi_x2 - roi_x
    roi_h = roi_y2 - roi_y

    if roi_w <= 0 or roi_h <= 0:
        return None

    conf = _confidence(mask, roi_x, roi_y, roi_w, roi_h)

    crop = gray[roi_y:roi_y2, roi_x:roi_x2]

    if crop.size == 0:
        return None

    # Use Otsu on the crop to get a local threshold for digit inference.
    if int(crop.min()) == int(crop.max()):
        local_threshold = brightness_threshold
    else:
        local_threshold, _ = cv2.threshold(
            crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        local_threshold = int(local_threshold)

    digit_boxes, decimal_box = _infer_digit_boxes(crop, local_threshold)

    roi = ROI(x=roi_x, y=roi_y, width=roi_w, height=roi_h)
    return LocatedDisplay(
        roi=roi,
        digit_boxes=digit_boxes,
        decimal_box=decimal_box,
        confidence=conf,
    )


def crop_display(image: np.ndarray, located: LocatedDisplay) -> np.ndarray:
    """Return the image crop of the located display ROI."""
    roi = located.roi
    return image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]


def crop_digits(image: np.ndarray, located: LocatedDisplay) -> np.ndarray:
    """Return a tight crop of just the digit row, excluding sub-display icons.

    Uses the digit_boxes y-extent (which is relative to the ROI) to cut out
    only the main character row, discarding indicators below it.
    """
    roi = located.roi
    boxes = located.digit_boxes
    if not boxes:
        return crop_display(image, located)
    y_min = min(b.y for b in boxes)
    y_max = max(b.y + b.height for b in boxes)
    return image[roi.y + y_min:roi.y + y_max, roi.x:roi.x + roi.width]


def located_decoder_config(
    located: LocatedDisplay,
    expected_format: str = "auto",
    segment_on_ratio: float = 0.15,
    decimal_on_ratio: float = 0.10,
) -> DecoderConfig:
    """Build a DecoderConfig from a LocatedDisplay.

    Lower on-ratios than the decoder default are used because auto-located
    boxes come from blurry camera frames where segment brightness is diluted.
    """
    return DecoderConfig(
        roi=located.roi,
        digit_boxes=located.digit_boxes,
        decimal_box=located.decimal_box,
        expected_format=expected_format,
        segment_on_ratio=segment_on_ratio,
        decimal_on_ratio=decimal_on_ratio,
    )
