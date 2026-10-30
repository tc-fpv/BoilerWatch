"""Reading a gauge frame: the DisplayReader contract and its live implementation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Callable, Protocol

import numpy as np

from boiler_reader.decoder import DecoderConfig, DigitBox, ROI, Reading, decode_frame
from boiler_reader.locator import crop_display, locate_display
from boiler_reader.ocr import read_display
from boiler_reader.settings import ViewerSettings


class DisplayReader(Protocol):
    """Turns a camera frame into a reading and the ROI it came from.

    Returns None when the display cannot be located.  May raise ValueError for
    an undecodable frame or ocr.LLMUnavailableError when the LLM is unreachable.
    """

    def read(self, image: np.ndarray) -> tuple[Reading, ROI] | None: ...


class ReaderState(Protocol):
    """The slice of application state a LiveDisplayReader needs."""

    settings: ViewerSettings
    reader_settings: dict[str, object]
    settings_lock: AbstractContextManager

    def get_roi(self) -> ROI | None: ...


def decoder_config(settings: ViewerSettings, roi: ROI) -> DecoderConfig:
    return DecoderConfig(
        roi=roi,
        digit_boxes=settings.digit_boxes or (DigitBox(0, 0, roi.width, roi.height),),
        decimal_box=settings.decimal_box,
        expected_format=settings.expected_format,
    )


class LiveDisplayReader:
    """Reads with the segment decoder when an ROI is fixed, else locates and asks the LLM."""

    def __init__(
        self,
        state: ReaderState,
        *,
        on_located: Callable[[dict[str, object], dict[str, object]], None] | None = None,
        on_prepared: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._state = state
        self._on_located = on_located
        self._on_prepared = on_prepared

    def read(self, image: np.ndarray) -> tuple[Reading, ROI] | None:
        state = self._state
        settings = state.settings
        roi = state.get_roi()
        if roi is not None and not (settings.auto_locate and settings.digit_boxes is None):
            return decode_frame(image, decoder_config(settings, roi)), roi
        with state.settings_lock:
            brightness = int(
                state.reader_settings.get("brightness_threshold", settings.brightness_threshold)
            )
            url = str(state.reader_settings.get("ollama_url", ""))
            model = str(state.reader_settings.get("ollama_model", "lfm3-vl:latest"))
        located = locate_display(image, brightness)
        if located is None:
            return None
        if self._on_located is not None:
            self._on_located(
                asdict(located.roi),
                {"width": image.shape[1], "height": image.shape[0]},
            )
        reading, _ = read_display(
            crop_display(image, located),
            url,
            model,
            on_prepared=self._on_prepared,
        )
        return reading, located.roi
