"""Decode a saved camera image for offline calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2

from boiler_reader.decoder import DecoderConfig, DigitBox, ROI, Reading, decode_frame


def _parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("coordinates must be integers in x,y,width,height order") from error
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("provide exactly four coordinates: x,y,width,height")
    if coordinates[0] < 0 or coordinates[1] < 0 or coordinates[2] <= 0 or coordinates[3] <= 0:
        raise argparse.ArgumentTypeError("origins must be non-negative and dimensions must be positive")
    return coordinates


def decode_image_file(image_path: str | Path, config: DecoderConfig) -> Reading:
    """Load a supported image file and pass it to the shared frame decoder."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read image file: {image_path}")
    return decode_frame(image, config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode a saved boiler-gauge image")
    parser.add_argument("image", type=Path, help="path to a JPEG or other OpenCV-readable image")
    parser.add_argument(
        "--roi",
        required=True,
        type=_parse_box,
        metavar="X,Y,W,H",
        help="gauge display crop in full-image coordinates",
    )
    parser.add_argument(
        "--digit-box",
        action="append",
        required=True,
        type=_parse_box,
        metavar="X,Y,W,H",
        help="one character box in ROI coordinates; repeat once per display position",
    )
    parser.add_argument(
        "--decimal-box",
        type=_parse_box,
        metavar="X,Y,W,H",
        help="decimal-point box in ROI coordinates",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "d", "d.d"),
        default="auto",
        help="expected display format (default: auto)",
    )
    parser.add_argument(
        "--invert-threshold",
        action="store_true",
        help="use for displays where the illuminated segments are darker than the background",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = DecoderConfig(
        roi=ROI(*args.roi),
        digit_boxes=tuple(DigitBox(*box) for box in args.digit_box),
        expected_format=args.format,
        decimal_box=DigitBox(*args.decimal_box) if args.decimal_box else None,
        invert_threshold=args.invert_threshold,
    )
    try:
        reading = decode_image_file(args.image, config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(asdict(reading), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())