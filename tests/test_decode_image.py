import cv2
import numpy as np
import pytest
from pathlib import Path

from boiler_reader.decode_image import decode_image_file
from boiler_reader.decoder import DecoderConfig, DigitBox, ROI


def test_decodes_jpeg_file(tmp_path) -> None:
    image = np.zeros((160, 120), dtype=np.uint8)
    box = DigitBox(10, 10, 100, 140)
    for x0, y0, x1, y1 in (
        (0.25, 0.08, 0.75, 0.18),
        (0.78, 0.20, 0.88, 0.46),
        (0.78, 0.54, 0.88, 0.80),
        (0.25, 0.82, 0.75, 0.92),
        (0.12, 0.54, 0.22, 0.80),
        (0.12, 0.20, 0.22, 0.46),
        (0.25, 0.46, 0.75, 0.56),
    ):
        left = box.x + int(box.width * x0) - 3
        top = box.y + int(box.height * y0) - 3
        right = box.x + int(box.width * x1) + 3
        bottom = box.y + int(box.height * y1) + 3
        cv2.rectangle(image, (left, top), (right, bottom), 255, -1)

    image_path = tmp_path / "gauge.jpg"
    assert cv2.imwrite(str(image_path), image)
    reading = decode_image_file(
        image_path,
        DecoderConfig(
            roi=ROI(0, 0, 120, 160),
            digit_boxes=(box,),
            expected_format="d",
            morphology_kernel_size=1,
        ),
    )

    assert reading.valid
    assert reading.value == 8.0
    assert reading.display_text == "8"


def test_rejects_unreadable_image_file(tmp_path) -> None:
    image_path = tmp_path / "missing.jpg"
    config = DecoderConfig(
        roi=ROI(0, 0, 120, 160),
        digit_boxes=(DigitBox(10, 10, 100, 140),),
        expected_format="d",
    )

    with pytest.raises(ValueError, match="could not read image file"):
        decode_image_file(image_path, config)


def test_decodes_supplied_boiler_image() -> None:
    image_path = Path(__file__).with_name("boiler.PNG")
    config = DecoderConfig(
        roi=ROI(330, 238, 110, 94),
        digit_boxes=(DigitBox(0, 0, 22, 85), DigitBox(30, 3, 48, 76)),
        decimal_box=DigitBox(13, 68, 9, 8),
    )

    reading = decode_image_file(image_path, config)

    assert reading.valid
    assert reading.value == 1.3
    assert reading.display_text == "13"


@pytest.mark.parametrize(
    ("image_name", "value", "unit", "display_text"),
    [
        ("boiler-pressure.jpg", 1.2, "bar", "12"),
        ("boiler-temperature.jpg", 52.3, "°C", "523"),
    ],
)
def test_decodes_pressure_and_temperature_with_shared_display_roi(
    image_name: str,
    value: float,
    unit: str,
    display_text: str,
) -> None:
    config = DecoderConfig(
        roi=ROI(255, 230, 170, 110),
        digit_boxes=(
            DigitBox(15, 10, 34, 69),
            DigitBox(53, 10, 49, 76),
            DigitBox(108, 15, 45, 76),
        ),
        decimal_box=DigitBox(88, 76, 12, 10),
    )

    reading = decode_image_file(Path(__file__).with_name("fixtures") / image_name, config)

    assert reading.valid
    assert reading.value == value
    assert reading.unit == unit
    assert reading.display_text == display_text