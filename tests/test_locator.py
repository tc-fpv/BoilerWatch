from pathlib import Path

import cv2
import numpy as np
import pytest

from boiler_reader.locator import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    LocatedDisplay,
    crop_display,
    locate_display,
)
from boiler_reader.ocr import read_display


def _fixture(name: str) -> Path:
    return Path(__file__).with_name("fixtures") / name


def test_locates_pressure_display_and_decodes_correctly() -> None:
    image = cv2.imread(str(_fixture("boiler-pressure.jpg")), cv2.IMREAD_UNCHANGED)
    assert image is not None

    result = locate_display(image)

    assert result is not None
    assert isinstance(result, LocatedDisplay)
    assert result.confidence > 0.5
    h, w = image.shape[:2]
    assert result.roi.width > 20
    assert result.roi.height > 10
    assert result.roi.x >= 0 and result.roi.y >= 0
    assert result.roi.x + result.roi.width <= w
    assert result.roi.y + result.roi.height <= h

    reading, _ = read_display(crop_display(image, result))
    assert reading.valid
    assert reading.value == pytest.approx(1.2, abs=0.1)
    assert reading.unit == "bar"


def test_locates_temperature_display_and_decodes_correctly() -> None:
    image = cv2.imread(str(_fixture("boiler-temperature.jpg")), cv2.IMREAD_UNCHANGED)
    assert image is not None

    result = locate_display(image)

    assert result is not None
    assert result.confidence > 0.5

    reading, _ = read_display(crop_display(image, result))
    assert reading.valid
    assert reading.value == pytest.approx(52.3, abs=0.1)
    assert reading.unit == "°C"


def test_returns_none_for_completely_black_frame() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = locate_display(image)

    assert result is None


def test_returns_none_when_only_osd_strip_is_bright() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[430:460, 50:600] = 200

    result = locate_display(image)

    assert result is None


def test_locates_display_in_grayscale_input() -> None:
    image = cv2.imread(str(_fixture("boiler-pressure.jpg")), cv2.IMREAD_GRAYSCALE)
    assert image is not None

    result = locate_display(image)

    assert result is not None
    assert result.confidence > 0.5


def test_locate_display_accepts_custom_threshold() -> None:
    image = cv2.imread(str(_fixture("boiler-pressure.jpg")), cv2.IMREAD_UNCHANGED)
    assert image is not None

    result_default = locate_display(image, brightness_threshold=DEFAULT_BRIGHTNESS_THRESHOLD)
    result_custom = locate_display(image, brightness_threshold=20)

    assert result_default is not None
    assert result_custom is not None
    for result in (result_default, result_custom):
        reading, _ = read_display(crop_display(image, result))
        assert reading.valid


def test_locates_moved_and_skewed_pressure_display() -> None:
    image = cv2.imread(str(_fixture("boiler-pressure moved.jpg")), cv2.IMREAD_UNCHANGED)
    assert image is not None

    result = locate_display(image)

    assert result is not None
    assert result.confidence > 0.5

    reading, _ = read_display(crop_display(image, result))
    assert reading.valid
    assert reading.value == pytest.approx(1.2, abs=0.1)
    assert reading.unit == "bar"


def test_crop_display_returns_roi_region() -> None:
    image = cv2.imread(str(_fixture("boiler-pressure.jpg")), cv2.IMREAD_UNCHANGED)
    assert image is not None

    result = locate_display(image)
    assert result is not None

    crop = crop_display(image, result)
    assert crop.shape[0] == result.roi.height
    assert crop.shape[1] == result.roi.width
