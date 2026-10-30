from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from boiler_reader import ocr
from boiler_reader.locator import crop_display, locate_display
from boiler_reader.ocr import _vote, read_display


def _fake_replies(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> list[np.ndarray]:
    """Make every LLM call return the next reply; returns the images sent."""
    sent: list[np.ndarray] = []
    queue = list(replies)

    def fake_ask(crop, url, model, timeout=60, processed=None):
        sent.append(processed)
        return queue.pop(0), 0.0

    monkeypatch.setattr(ocr, "_ask_ollama", fake_ask)
    return sent


@pytest.fixture
def crop() -> np.ndarray:
    # Dim digit-like blob; content is irrelevant because replies are faked.
    img = np.zeros((40, 80), dtype=np.uint8)
    img[5:35, 10:20] = 120
    img[5:35, 50:60] = 200
    return img


def test_vote_clear_majority() -> None:
    assert _vote(["11", "21", "11", "11"]) == ("11", 3)


def test_vote_tie_has_no_winner() -> None:
    assert _vote(["21", "88", "11", "21", "88", "11"])[0] is None


def test_all_three_agree_is_confident_and_needs_no_tiebreak(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    sent = _fake_replies(monkeypatch, ["11", "11", "11"])

    reading, _ = read_display(crop, "http://unused")

    assert len(sent) == 3
    assert reading.valid and reading.value == pytest.approx(1.1) and reading.confidence == 1.0


def test_disagreement_adds_a_fourth_read_and_takes_the_majority(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    sent = _fake_replies(monkeypatch, ["21", "11", "11", "11"])

    reading, _ = read_display(crop, "http://unused")

    assert len(sent) == 4
    assert reading.valid and reading.value == pytest.approx(1.1)
    assert reading.confidence == pytest.approx(0.75)


def test_no_majority_is_invalid_and_lists_the_candidates(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    _fake_replies(monkeypatch, ["21", "88", "11", "12"])

    reading, _ = read_display(crop, "http://unused")

    assert not reading.valid and reading.value is None
    assert reading.raw_pattern == "11/12/21/88"


def test_a_failed_request_does_not_abort_the_vote(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    calls = iter([RuntimeError("boom"), "11", "11", "11"])

    def fake_ask(crop, url, model, timeout=60, processed=None):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item, 0.0

    monkeypatch.setattr(ocr, "_ask_ollama", fake_ask)

    reading, _ = read_display(crop, "http://unused")

    assert reading.valid and reading.value == pytest.approx(1.1)


def test_each_preprocessing_produces_a_distinct_image(crop: np.ndarray) -> None:
    images = [p(crop) for p in ocr._PRIMARY_PREPROCESSORS] + [ocr._TIEBREAK_PREPROCESSOR(crop)]
    for img in images:
        assert img.dtype == np.uint8 and img.ndim == 2
    # Compare without the random corner pixel.
    stripped = [i.copy() for i in images]
    for i in stripped:
        i[-1, -1] = 0
    for a in range(len(stripped)):
        for b in range(a + 1, len(stripped)):
            assert stripped[a].shape != stripped[b].shape or not np.array_equal(stripped[a], stripped[b])


@pytest.mark.parametrize(
    ("name", "value", "unit"),
    [
        ("boiler-1.1 dim read as colon-1.jpg", 1.1, "bar"),
        ("boiler-1.1 mixed up as 1.2.jpg", 1.1, "bar"),
        ("boiler-1.5 being read as 5.jpg", 1.5, "bar"),
        ("boiler-32.1C.PNG", 32.1, "°C"),
    ],
)
def test_fixture_reads_with_vote(name: str, value: float, unit: str) -> None:
    # Needs the Ollama server in OLLAMA_URL, like the other fixture tests.
    path = Path(__file__).with_name("fixtures") / name
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    located = locate_display(image)
    assert located is not None

    reading, _ = read_display(crop_display(image, located))

    assert reading.valid
    assert reading.value == pytest.approx(value, abs=0.01)
    assert reading.unit == unit


_LIVE_FRAMES = sorted(Path(__file__).with_name("fixtures").glob("boiler-1.1-live-*.jpg"))


@pytest.mark.parametrize("path", _LIVE_FRAMES, ids=lambda p: p.stem)
def test_live_frames_of_1_1_bar_read_correctly_with_vote(path: Path) -> None:
    # 16 live frames of a steady 1.1 bar; single-pass reads gave 2.1 or ":" on most.
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    located = locate_display(image)
    assert located is not None

    reading, _ = read_display(crop_display(image, located))

    assert reading.valid
    assert reading.value == pytest.approx(1.1, abs=0.01)
    assert reading.unit == "bar"


def test_every_request_failing_raises_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    def fake_ask(crop, url, model, timeout=60, processed=None):
        raise OSError("connection refused")

    monkeypatch.setattr(ocr, "_ask_ollama", fake_ask)

    with pytest.raises(ocr.LLMUnavailableError):
        read_display(crop, "http://unused")
