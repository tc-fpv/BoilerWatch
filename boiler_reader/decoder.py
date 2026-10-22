"""Image preprocessing and seven-segment display decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DigitBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DecoderConfig:
    roi: ROI
    digit_boxes: tuple[DigitBox, ...]
    expected_format: str = "auto"
    decimal_box: DigitBox | None = None
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    fallback_threshold: int = 127
    morphology_kernel_size: int = 2
    segment_on_ratio: float = 0.5
    decimal_on_ratio: float = 0.2
    invert_threshold: bool = False


@dataclass(frozen=True)
class Reading:
    value: float | None
    valid: bool
    raw_pattern: str
    confidence: float
    display_text: str | None = None
    error_code: str | None = None


SEGMENT_ZONES: Final[dict[str, tuple[float, float, float, float]]] = {
    "a": (0.25, 0.08, 0.75, 0.18),
    "b": (0.78, 0.20, 0.88, 0.46),
    "c": (0.78, 0.54, 0.88, 0.80),
    "d": (0.25, 0.82, 0.75, 0.92),
    "e": (0.12, 0.54, 0.22, 0.80),
    "f": (0.12, 0.20, 0.22, 0.46),
    "g": (0.25, 0.46, 0.75, 0.56),
}

_DIGIT_PATTERNS: Final[dict[str, str]] = {
    "1111110": "0",
    "0110000": "1",
    "1101101": "2",
    "1111001": "3",
    "0110011": "4",
    "1011011": "5",
    "1011111": "6",
    "1110000": "7",
    "1111111": "8",
    "1111011": "9",
}

_LETTER_PATTERNS: Final[dict[str, str]] = {
    "1110111": "A",
    "0011111": "b",
    "1001110": "C",
    "0111101": "d",
    "1001111": "E",
    "1000111": "F",
    "0110111": "H",
    "0001110": "L",
    "0010101": "n",
    "0011101": "o",
    "1100111": "P",
    "0000101": "r",
    "0111110": "U",
    "0000001": "-",
}

_BLANK_PATTERN: Final[str] = "0000000"

KNOWN_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "A10", "A11", "A20", "A21", "A30", "A40", "A41", "A42", "A58",
        "A59", "A60", "A70", "A71", "A80", "A81", "A90", "A91", "ECS",
        "SES", "FILL",
    }
)

_ERROR_CODE_CHARACTER_PATTERNS: Final[dict[str, str]] = {
    **{character: pattern for pattern, character in _DIGIT_PATTERNS.items()},
    **{character: pattern for pattern, character in _LETTER_PATTERNS.items()},
    "S": "1011011",
    "I": "0110000",
}

ERROR_CODE_PATTERNS: Final[dict[tuple[str, ...], str]] = {}
for _error_code in KNOWN_ERROR_CODES:
    _display_aliases = ("FIL",) if _error_code == "FILL" else (_error_code,)
    for _display_alias in _display_aliases:
        if all(character in _ERROR_CODE_CHARACTER_PATTERNS for character in _display_alias):
            _display_pattern = tuple(
                _ERROR_CODE_CHARACTER_PATTERNS[character]
                for character in _display_alias
            )
            ERROR_CODE_PATTERNS[_display_pattern] = _error_code


def _validate_config(config: DecoderConfig) -> None:
    if not config.digit_boxes:
        raise ValueError("at least one digit box is required")
    if len(config.digit_boxes) > 3:
        raise ValueError("at most three character boxes are supported")
    allowed_formats = {"auto", "d", "d.d"}
    if config.expected_format not in allowed_formats:
        raise ValueError("expected_format must be 'auto', 'd', or 'd.d'")
    if config.expected_format != "auto" and config.expected_format.count("d") > len(config.digit_boxes):
        raise ValueError("expected_format digit count must match digit_boxes")
    if config.expected_format in {"d.d", "dd.d"} and config.decimal_box is None:
        raise ValueError("decimal_box is required for a pressure format")
    if not 0 < config.segment_on_ratio < 1 or not 0 < config.decimal_on_ratio < 1:
        raise ValueError("segment and decimal on-ratios must be between 0 and 1")
    if not 0 <= config.fallback_threshold <= 255:
        raise ValueError("fallback_threshold must be between 0 and 255")
    if config.morphology_kernel_size < 1:
        raise ValueError("morphology_kernel_size must be positive")


def _crop(image: np.ndarray, box: ROI | DigitBox) -> np.ndarray:
    height, width = image.shape[:2]
    if box.x < 0 or box.y < 0 or box.width <= 0 or box.height <= 0:
        raise ValueError("image boxes must have non-negative origins and positive dimensions")
    if box.x + box.width > width or box.y + box.height > height:
        raise ValueError("image box extends beyond its image")
    return image[box.y : box.y + box.height, box.x : box.x + box.width]


def preprocess_roi(image: np.ndarray, config: DecoderConfig) -> tuple[np.ndarray, int]:
    """Crop, enhance, threshold, and lightly clean the configured ROI."""
    roi = _crop(image, config.roi)
    if roi.ndim == 2:
        grayscale = roi
    elif roi.ndim == 3 and roi.shape[2] == 3:
        grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    elif roi.ndim == 3 and roi.shape[2] == 4:
        grayscale = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA")

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
    )
    enhanced = clahe.apply(grayscale)
    if int(enhanced.min()) == int(enhanced.max()):
        threshold = config.fallback_threshold
    else:
        threshold, _ = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        threshold = int(threshold)

    threshold_type = cv2.THRESH_BINARY_INV if config.invert_threshold else cv2.THRESH_BINARY
    _, binary = cv2.threshold(enhanced, threshold, 255, threshold_type)
    kernel_size = config.morphology_kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary, threshold


def _normalized_crop(image: np.ndarray, box: DigitBox, zone: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = zone
    left = box.x + int(box.width * x0)
    top = box.y + int(box.height * y0)
    right = box.x + max(int(box.width * x1), int(box.width * x0) + 1)
    bottom = box.y + max(int(box.height * y1), int(box.height * y0) + 1)
    return image[top:bottom, left:right]


def _sample_digit(binary: np.ndarray, box: DigitBox, on_ratio: float) -> tuple[str, list[float]]:
    samples: list[float] = []
    for zone in SEGMENT_ZONES.values():
        pixels = _normalized_crop(binary, box, zone)
        samples.append(float(np.count_nonzero(pixels) / pixels.size))
    pattern = "".join("1" if sample >= on_ratio else "0" for sample in samples)
    return pattern, samples


def _sample_decimal(binary: np.ndarray, box: DigitBox, on_ratio: float) -> tuple[bool, float]:
    pixels = _crop(binary, box)
    ratio = float(np.count_nonzero(pixels) / pixels.size)
    return ratio >= on_ratio, ratio


def _sample_confidence(samples: list[float], cutoff: float) -> float:
    if not samples:
        return 0.0
    return float(np.mean([min(1.0, 2.0 * abs(sample - cutoff)) for sample in samples]))


def _decode_symbol(pattern: str) -> str | None:
    if pattern == _BLANK_PATTERN:
        return " "
    return _DIGIT_PATTERNS.get(pattern) or _LETTER_PATTERNS.get(pattern) or "?"


def decode_frame(image: np.ndarray, config: DecoderConfig) -> Reading:
    """Decode configured digit boxes from a camera frame."""
    _validate_config(config)
    binary, _ = preprocess_roi(image, config)
    patterns: list[str] = []
    confidence_samples: list[float] = []
    for box in config.digit_boxes:
        _crop(binary, box)
        pattern, samples = _sample_digit(binary, box, config.segment_on_ratio)
        patterns.append(pattern)
        confidence_samples.extend(samples)

    decimal_on = False
    if config.decimal_box is not None:
        decimal_on, decimal_ratio = _sample_decimal(
            binary, config.decimal_box, config.decimal_on_ratio
        )
        confidence_samples.append(decimal_ratio)

    raw_pattern = "/".join(patterns)
    if config.decimal_box is not None:
        raw_pattern += f";dp={int(decimal_on)}"

    decoded_symbols = [_decode_symbol(pattern) for pattern in patterns]
    error_code = ERROR_CODE_PATTERNS.get(tuple(patterns))
    display_text = (
        "".join(symbol for symbol in decoded_symbols if symbol is not None).strip()
        if all(symbol is not None for symbol in decoded_symbols)
        else None
    )
    if error_code is not None:
        display_text = "FIL" if error_code == "FILL" else error_code
    digit_symbols = [symbol for symbol in decoded_symbols if symbol not in (None, " ")]
    blank_count = decoded_symbols.count(" ")
    numeric_symbols = all(
        symbol is not None and symbol.isdigit()
        for symbol in decoded_symbols
        if symbol != " "
    )
    allowed_blank_layout = blank_count == 0 or (
        len(patterns) == 3 and blank_count == 1 and decoded_symbols[0] == " "
    )
    required_digits = {"d": 1, "d.d": 2}.get(config.expected_format)
    numeric_format = len(digit_symbols)
    single_digit_mode = config.expected_format == "d"
    valid = numeric_symbols and allowed_blank_layout and (
        (single_digit_mode and not decimal_on and numeric_format == 1)
        or (
            not single_digit_mode
            and decimal_on
            and numeric_format == 2
            and (required_digits is None or numeric_format == required_digits)
        )
    )
    value: float | None = None
    if valid:
        rendered_digits = "".join(digit_symbols)
        if single_digit_mode:
            value = float(rendered_digits)
        else:
            value = float(f"{rendered_digits[:-1]}.{rendered_digits[-1]}")

    return Reading(
        value=value,
        valid=valid,
        raw_pattern=raw_pattern,
        confidence=_sample_confidence(
            confidence_samples,
            config.segment_on_ratio,
        ),
        display_text=display_text,
        error_code=error_code,
    )