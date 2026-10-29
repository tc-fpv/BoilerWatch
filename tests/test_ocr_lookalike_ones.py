from pathlib import Path

import cv2
import pytest

from boiler_reader.locator import crop_display, locate_display
from boiler_reader.ocr import _interpret, _normalise_lookalike_ones, read_display


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (":1", "11"),
        ("!1", "11"),
        ("|5", "15"),
        ("l2", "12"),
        ("32.:", "32.1"),
        ("12", "12"),
        ("321", "321"),
    ],
)
def test_lookalike_one_is_read_as_digit(reply: str, expected: str) -> None:
    assert _normalise_lookalike_ones(reply) == expected


@pytest.mark.parametrize(
    "reply",
    ["ECS", "FIL", "A10", "SES", "Digits: 11", "I cannot read this", ":", ""],
)
def test_other_replies_are_left_untouched(reply: str) -> None:
    assert _normalise_lookalike_ones(reply) == reply


def test_colon_one_reply_interprets_as_1_1_bar() -> None:
    reading = _interpret(_normalise_lookalike_ones(":1"))

    assert reading.valid
    assert reading.value == pytest.approx(1.1)
    assert reading.unit == "bar"


def test_dim_frame_with_one_beside_decimal_reads_1_1_bar() -> None:
    # Live frame: the first "1" sits above its decimal point and the model
    # replied ":1" every time, which was rejected as invalid.
    path = Path(__file__).with_name("fixtures") / "boiler-1.1 dim read as colon-1.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None

    located = locate_display(image)
    assert located is not None

    reading, _ = read_display(crop_display(image, located))

    assert reading.valid
    assert reading.value == pytest.approx(1.1, abs=0.01)
    assert reading.unit == "bar"
