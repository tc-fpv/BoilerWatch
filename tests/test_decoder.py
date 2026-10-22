import cv2
import numpy as np
import pytest

from boiler_reader.decoder import (
    SEGMENT_ZONES,
    DecoderConfig,
    DigitBox,
    ERROR_CODE_PATTERNS,
    KNOWN_ERROR_CODES,
    ROI,
    decode_frame,
)


_DIGIT_PATTERNS = {
    "0": "1111110",
    "1": "0110000",
    "2": "1101101",
    "3": "1111001",
    "4": "0110011",
    "5": "1011011",
    "6": "1011111",
    "7": "1110000",
    "8": "1111111",
    "9": "1111011",
}

_LETTER_PATTERNS = {
    "A": "1110111",
    "b": "0011111",
    "C": "1001110",
    "E": "1001111",
    "F": "1000111",
    "H": "0110111",
    "L": "0001110",
    "P": "1100111",
    "r": "0000101",
    "U": "0111110",
    "-": "0000001",
}


def _segment_bounds(box: DigitBox, zone: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = zone
    pad_x = max(2, int(box.width * 0.025))
    pad_y = max(2, int(box.height * 0.025))
    left = box.x + int(box.width * x0) - pad_x
    top = box.y + int(box.height * y0) - pad_y
    right = box.x + int(box.width * x1) + pad_x
    bottom = box.y + int(box.height * y1) + pad_y
    return left, top, right, bottom


def _render(pattern: str, decimal: bool = False) -> np.ndarray:
    image = np.zeros((160, 120), dtype=np.uint8)
    box = DigitBox(10, 10, 100, 140)
    for lit, zone in zip(pattern, SEGMENT_ZONES.values(), strict=True):
        if lit == "1":
            left, top, right, bottom = _segment_bounds(box, zone)
            cv2.rectangle(image, (left, top), (right, bottom), 255, -1)
    if decimal:
        cv2.circle(image, (108, 143), 4, 255, -1)
    return image


def _render_characters(patterns: tuple[str, ...], decimal: bool = False) -> np.ndarray:
    boxes = tuple(DigitBox(10 + index * 120, 10, 100, 140) for index in range(len(patterns)))
    image = np.zeros((160, len(patterns) * 120), dtype=np.uint8)
    for box, pattern in zip(boxes, patterns, strict=True):
        for lit, zone in zip(pattern, SEGMENT_ZONES.values(), strict=True):
            if lit == "1":
                left, top, right, bottom = _segment_bounds(box, zone)
                cv2.rectangle(image, (left, top), (right, bottom), 255, -1)
    if decimal:
        cv2.circle(image, (115, 143), 4, 255, -1)
    return image


def _single_digit_config() -> DecoderConfig:
    return DecoderConfig(
        roi=ROI(0, 0, 120, 160),
        digit_boxes=(DigitBox(10, 10, 100, 140),),
        expected_format="d",
        morphology_kernel_size=1,
    )


@pytest.mark.parametrize(("digit", "pattern"), _DIGIT_PATTERNS.items())
def test_decodes_each_digit(digit: str, pattern: str) -> None:
    reading = decode_frame(_render(pattern), _single_digit_config())

    assert reading.valid
    assert reading.value == float(digit)
    assert reading.raw_pattern == pattern
    assert reading.confidence > 0.5


@pytest.mark.parametrize("pattern", ["0000000", "1001001"])
def test_unmapped_patterns_are_invalid(pattern: str) -> None:
    reading = decode_frame(_render(pattern), _single_digit_config())

    assert not reading.valid
    assert reading.value is None
    assert reading.raw_pattern == pattern


def test_decodes_decimal_format_and_requires_decimal_point() -> None:
    first_box = DigitBox(10, 10, 100, 140)
    second_box = DigitBox(180, 10, 100, 140)
    image = np.zeros((160, 320), dtype=np.uint8)
    for box, pattern in ((first_box, _DIGIT_PATTERNS["1"]), (second_box, _DIGIT_PATTERNS["5"])):
        for lit, zone in zip(pattern, SEGMENT_ZONES.values(), strict=True):
            if lit == "1":
                left, top, right, bottom = _segment_bounds(box, zone)
                cv2.rectangle(image, (left, top), (right, bottom), 255, -1)
    decimal_box = DigitBox(128, 133, 12, 12)
    cv2.rectangle(image, (131, 136), (137, 142), 255, -1)
    config = DecoderConfig(
        roi=ROI(0, 0, 320, 160),
        digit_boxes=(first_box, second_box),
        expected_format="d.d",
        decimal_box=decimal_box,
        morphology_kernel_size=1,
    )

    reading = decode_frame(image, config)
    without_point = decode_frame(np.zeros_like(image), config)

    assert reading.valid
    assert reading.value == 1.5
    assert reading.raw_pattern.endswith(";dp=1")
    assert not without_point.valid
    assert without_point.value is None


@pytest.mark.parametrize("expected_format", ["auto", "dd.d"])
def test_decodes_temperature_format(expected_format: str) -> None:
    patterns = (
        _DIGIT_PATTERNS["6"],
        _DIGIT_PATTERNS["5"],
        _DIGIT_PATTERNS["3"],
    )
    boxes = tuple(DigitBox(10 + index * 120, 10, 100, 140) for index in range(3))
    config = DecoderConfig(
        roi=ROI(0, 0, 360, 160),
        digit_boxes=boxes,
        expected_format=expected_format,
        decimal_box=DigitBox(108, 136, 14, 12),
        morphology_kernel_size=1,
    )

    reading = decode_frame(_render_characters(patterns, decimal=True), config)

    assert reading.valid
    assert reading.value == 65.3
    assert reading.unit == "°C"
    assert reading.display_text == "653"


@pytest.mark.parametrize(
    ("text", "patterns"),
    [
        ("E01", (_LETTER_PATTERNS["E"], _DIGIT_PATTERNS["0"], _DIGIT_PATTERNS["1"])),
        ("Err", (_LETTER_PATTERNS["E"], _LETTER_PATTERNS["r"], _LETTER_PATTERNS["r"])),
    ],
)
def test_decodes_three_character_error_code(text: str, patterns: tuple[str, ...]) -> None:
    boxes = tuple(DigitBox(10 + index * 120, 10, 100, 140) for index in range(3))
    config = DecoderConfig(
        roi=ROI(0, 0, 360, 160),
        digit_boxes=boxes,
        morphology_kernel_size=1,
    )

    reading = decode_frame(_render_characters(patterns), config)

    assert not reading.valid
    assert reading.value is None
    assert reading.display_text == text


def test_decodes_error_code_with_shared_display_geometry() -> None:
    boxes = (
        DigitBox(15, 10, 34, 69),
        DigitBox(53, 10, 49, 76),
        DigitBox(108, 15, 45, 76),
    )
    image = np.zeros((110, 170), dtype=np.uint8)
    patterns = (
        _LETTER_PATTERNS["A"],
        _DIGIT_PATTERNS["1"],
        _DIGIT_PATTERNS["0"],
    )
    for box, pattern in zip(boxes, patterns, strict=True):
        for lit, zone in zip(pattern, SEGMENT_ZONES.values(), strict=True):
            if lit == "1":
                left, top, right, bottom = _segment_bounds(box, zone)
                cv2.rectangle(image, (left, top), (right, bottom), 255, -1)
    config = DecoderConfig(
        roi=ROI(0, 0, 170, 110),
        digit_boxes=boxes,
        decimal_box=DigitBox(88, 76, 12, 10),
        morphology_kernel_size=1,
    )

    reading = decode_frame(image, config)

    assert not reading.valid
    assert reading.value is None
    assert reading.display_text == "A10"
    assert reading.error_code == "A10"


def test_unknown_glyph_does_not_form_a_pressure_reading() -> None:
    patterns = ("1001001", _DIGIT_PATTERNS["2"], _DIGIT_PATTERNS["3"])
    boxes = tuple(DigitBox(10 + index * 120, 10, 100, 140) for index in range(3))
    config = DecoderConfig(
        roi=ROI(0, 0, 360, 160),
        digit_boxes=boxes,
        morphology_kernel_size=1,
    )

    reading = decode_frame(_render_characters(patterns, decimal=True), config)

    assert not reading.valid
    assert reading.value is None
    assert reading.display_text == "?23"


def test_three_populated_characters_without_decimal_are_invalid() -> None:
    patterns = (_DIGIT_PATTERNS["1"], _DIGIT_PATTERNS["2"], _DIGIT_PATTERNS["3"])
    boxes = tuple(DigitBox(10 + index * 120, 10, 100, 140) for index in range(3))
    config = DecoderConfig(
        roi=ROI(0, 0, 360, 160),
        digit_boxes=boxes,
        decimal_box=DigitBox(108, 136, 14, 12),
        morphology_kernel_size=1,
    )

    reading = decode_frame(_render_characters(patterns), config)

    assert not reading.valid
    assert reading.value is None
    assert reading.unit is None
    assert reading.display_text == "123"


@pytest.mark.parametrize(
    ("patterns", "expected_code"),
    [
        (patterns, code)
        for patterns, code in ERROR_CODE_PATTERNS.items()
    ],
)
def test_recognizes_listed_manual_error_codes(
    patterns: tuple[str, ...],
    expected_code: str,
) -> None:
    boxes = tuple(DigitBox(10 + index * 120, 10, 100, 140) for index in range(len(patterns)))
    config = DecoderConfig(
        roi=ROI(0, 0, len(patterns) * 120, 160),
        digit_boxes=boxes,
        morphology_kernel_size=1,
    )

    reading = decode_frame(_render_characters(patterns), config)

    assert not reading.valid
    assert reading.value is None
    assert reading.error_code == expected_code


def test_error_code_catalog_includes_manual_entries() -> None:
    assert KNOWN_ERROR_CODES == {
        "A10", "A11", "A20", "A21", "A30", "A40", "A41", "A42", "A58",
        "A59", "A60", "A70", "A71", "A80", "A81", "A90", "A91", "ECS",
        "SES", "FILL",
    }


def test_decodes_pressure_with_blank_leading_character() -> None:
    patterns = ("0000000", _DIGIT_PATTERNS["1"], _DIGIT_PATTERNS["5"])
    boxes = tuple(DigitBox(10 + index * 120, 10, 100, 140) for index in range(3))
    config = DecoderConfig(
        roi=ROI(0, 0, 360, 160),
        digit_boxes=boxes,
        decimal_box=DigitBox(108, 136, 14, 12),
        morphology_kernel_size=1,
    )

    reading = decode_frame(_render_characters(patterns, decimal=True), config)

    assert reading.valid
    assert reading.value == 1.5
    assert reading.display_text == "15"


def test_rejects_digit_boxes_outside_roi() -> None:
    config = DecoderConfig(
        roi=ROI(0, 0, 120, 160),
        digit_boxes=(DigitBox(30, 10, 100, 140),),
        expected_format="d",
        morphology_kernel_size=1,
    )

    with pytest.raises(ValueError, match="extends beyond"):
        decode_frame(np.zeros((160, 120), dtype=np.uint8), config)