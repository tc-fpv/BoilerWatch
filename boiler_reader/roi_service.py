"""Setting the display region of interest by hand and reading it straight away."""

from __future__ import annotations

from dataclasses import asdict, replace

from boiler_reader.analysis import FrameAnalyzer
from boiler_reader.decoder import ROI, decode_frame
from boiler_reader.frames import annotate_roi, load_frame
from boiler_reader.locator import locate_display, located_decoder_config
from boiler_reader.reader import decoder_config
from boiler_reader.settings import persist_json
from boiler_reader.state import AppState


class RoiService:
    """Raises ValueError/KeyError/TypeError for bad input and OSError if saving fails."""

    def __init__(self, state: AppState, analyzer: FrameAnalyzer) -> None:
        self._state = state
        self._analyzer = analyzer

    def set_roi(self, payload: dict[str, object]) -> dict[str, object] | None:
        """Adopt the ROI, remember it and return the UI result; None if there is no frame."""
        state = self._state
        settings = state.settings
        roi = ROI(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )
        image = load_frame(settings.image_path)
        if image is None:
            return None
        annotate_roi(image, roi)  # validates the ROI against the frame
        if settings.digit_boxes is not None:
            config = decoder_config(settings, roi)
        else:
            located = locate_display(image, settings.brightness_threshold)
            if located is not None:
                config = located_decoder_config(
                    replace(located, roi=roi), settings.expected_format
                )
            else:
                config = decoder_config(settings, roi)
        reading = decode_frame(image, config)
        if settings.roi_state_path is not None:
            persist_json(settings.roi_state_path, asdict(roi))
        state.set_roi(roi)
        return self._analyzer.build_result(reading, image, roi)
