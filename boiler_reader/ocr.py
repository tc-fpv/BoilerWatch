"""EasyOCR-based reading of the located seven-segment display crop.

The decimal point is intentionally not read — EasyOCR drops it reliably on
seven-segment displays.  Format is inferred from digit count:
  2 digits → pressure  (e.g. "12" → 1.2 bar)
  3 digits → temperature or error code  (e.g. "523" → 52.3 °C, "ECS" → error)
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import cv2
import numpy as np

from boiler_reader.decoder import KNOWN_ERROR_CODES, Reading

if TYPE_CHECKING:
    import easyocr as _easyocr_module


# Characters that can appear on this boiler display.
_ALLOWLIST = "0123456789ABCEFHLPSUbdnor-"

# Upscale factor applied to the crop before OCR — improves recognition of
# small, blurry segment glyphs.
_UPSCALE = 4


@lru_cache(maxsize=1)
def _reader() -> "_easyocr_module.Reader":
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Upscale and invert the display crop for OCR.

    The bottom ~20% of the crop contains the decimal-point indicator and
    sub-display icons.  Cropping it out prevents EasyOCR from reading the
    decimal dot as a '0' or other character.
    """
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    h = gray.shape[0]
    gray = gray[:int(h * 0.80), :]
    large = cv2.resize(gray, None, fx=_UPSCALE, fy=_UPSCALE, interpolation=cv2.INTER_CUBIC)
    return cv2.bitwise_not(large)


def _read_crop(crop: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Run EasyOCR on a preprocessed crop, return (raw_text, confidence, processed_image)."""
    processed = _preprocess_crop(crop)
    results = _reader().readtext(processed, allowlist=_ALLOWLIST, detail=1)
    if not results:
        return "", 0.0, processed
    # Sort left-to-right and join — handles multi-detection splits.
    results.sort(key=lambda r: r[0][0][0])
    text = "".join(r[1] for r in results).strip()
    confidence = float(np.mean([r[2] for r in results]))
    return text, confidence, processed


def _interpret(raw: str, confidence: float) -> Reading:
    """Turn a raw OCR string into a Reading using digit-count format rules."""
    # Normalise: strip spaces and dots (decimal dropped by EasyOCR).
    cleaned = raw.replace(" ", "").replace(".", "")

    n = len(cleaned)

    # Three-character string — check error codes first.
    if n == 3:
        upper = cleaned.upper()
        # FIL on display = FILL error code.
        code = "FILL" if upper == "FIL" else upper if upper in KNOWN_ERROR_CODES else None
        if code is not None:
            return Reading(
                value=None,
                valid=False,
                raw_pattern=raw,
                confidence=confidence,
                display_text=cleaned,
                error_code=code,
                unit=None,
            )
        # Not an error code — try temperature (nn.n).
        if cleaned.isdigit():
            value = float(f"{cleaned[:2]}.{cleaned[2]}")
            return Reading(
                value=value,
                valid=True,
                raw_pattern=raw,
                confidence=confidence,
                display_text=cleaned,
                error_code=None,
                unit="°C",
            )

    # Two-character string — pressure (n.n).
    if n == 2 and cleaned.isdigit():
        value = float(f"{cleaned[0]}.{cleaned[1]}")
        return Reading(
            value=value,
            valid=True,
            raw_pattern=raw,
            confidence=confidence,
            display_text=cleaned,
            error_code=None,
            unit="bar",
        )

    # Anything else is invalid.
    return Reading(
        value=None,
        valid=False,
        raw_pattern=raw,
        confidence=confidence,
        display_text=cleaned or None,
        error_code=None,
        unit=None,
    )


def read_display(crop: np.ndarray) -> tuple[Reading, np.ndarray]:
    """Read a located display crop and return (Reading, preprocessed_image).

    crop should be a grayscale or BGR image of just the digit row,
    as returned by the locator.
    """
    raw, confidence, processed = _read_crop(crop)
    reading = _interpret(raw, confidence)
    print(
        f"ocr: raw={repr(raw)} -> "
        f"{'valid' if reading.valid else 'invalid'} "
        f"{reading.value} {reading.unit or ''} "
        f"(conf={confidence:.2f})"
    )
    return reading, processed
