"""LLM-based reading of the located seven-segment display crop.

Sends differently preprocessed copies of the crop to a local Ollama VLM
(default: lfm3-vl:latest) and takes a majority vote over the replies.  Each
preprocessing fails on a different scene, so they rarely fail together.  The
returned digits are interpreted using digit-count format rules:
  2 digits → pressure  (e.g. "12" → 1.2 bar)
  3 digits → temperature or error code  (e.g. "321" → 32.1 °C, "ECS" → error)
"""

from __future__ import annotations

import base64
import collections
import dataclasses
import json
import logging
import os
import random
import re
import time
import urllib.request
from typing import Callable

import cv2
import numpy as np

from boiler_reader.decoder import KNOWN_ERROR_CODES, Reading


_UPSCALE = 4


class LLMUnavailableError(RuntimeError):
    """Every request to the LLM server failed, so there is no reading to judge."""


def _log(event: str, **fields: object) -> None:
    logging.getLogger("boiler_reader.ocr").info(
        json.dumps({"event": event, **fields}, separators=(",", ":"))
    )

_PROMPT = (
    "This is a cropped, upscaled seven-segment display from a boiler gauge. "
    "The digits are bright white on a dark background. "
    "Read the digits shown. "
    "Reply with only the digits, e.g. '12' or '321'. "
    "Do not include decimal points or units. Do not explain or add any other text. "
    "Step 1: read the digits. "
    "Step 2: if you found fewer than 2 digits, look again for a thin vertical "
    "stroke (the digit 1), usually at the left, and include it. "
    "The final answer is read left to right."
)


_ONE_LOOKALIKES = ":!|lIi"
_DIGITS_AND_LOOKALIKES = re.compile(r"[\d.\s" + re.escape(_ONE_LOOKALIKES) + r"]+")


def _normalise_lookalike_ones(raw: str) -> str:
    """Read ``:``/``!``/``|``/``l``/``I`` as the digit 1 in a bare digit reply.

    A seven-segment 1 lights only the right-hand segments, so beside its
    decimal point it looks like ``!`` or ``:`` and the model returns e.g. ``:1``
    for 1.1.  Applied only when the whole reply is digits plus lookalikes, so
    explanation text and letter codes (ECS, FIL) are left untouched.
    """
    text = raw.strip()
    if not (_DIGITS_AND_LOOKALIKES.fullmatch(text) and any(c.isdigit() for c in text)):
        return raw
    return re.sub("[" + re.escape(_ONE_LOOKALIKES) + "]", "1", text)


def _to_gray(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop


def _upscale(gray: np.ndarray) -> np.ndarray:
    """4x upscale, then write a random-value pixel to the bottom-right corner.

    The pixel makes each call produce unique image bytes, preventing Ollama's
    KV cache from replaying a stale inference for an identical-looking image.
    """
    upscaled = cv2.resize(gray, None, fx=_UPSCALE, fy=_UPSCALE, interpolation=cv2.INTER_CUBIC)
    upscaled[-1, -1] = random.randint(0, 255)
    return upscaled


def _otsu(gray: np.ndarray) -> tuple[float, np.ndarray]:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


def _preprocess_crop_for_llm(crop: np.ndarray) -> np.ndarray:
    """Grayscale + 4x upscale, bright segments on dark background."""
    return _upscale(_to_gray(crop))


def _preprocess_saturated(crop: np.ndarray) -> np.ndarray:
    """Scale brightness so 0.75 x the Otsu level becomes white.

    Lifts dim live frames so a thin "1" is as bright as the other segments.
    """
    gray = _to_gray(crop)
    level, _ = _otsu(gray)
    scaled = gray.astype(np.float32) * 255 / max(0.75 * level, 1)
    return _upscale(np.clip(scaled, 0, 255).astype(np.uint8))


def _preprocess_binary_padded(crop: np.ndarray) -> np.ndarray:
    """Otsu black-and-white with a black border so edge digits are not clipped."""
    _, binary = _otsu(_to_gray(crop))
    padded = cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
    return _upscale(padded)


def _preprocess_binary_inverted(crop: np.ndarray) -> np.ndarray:
    """Otsu black-and-white, inverted (dark digits on white)."""
    _, binary = _otsu(_to_gray(crop))
    return _upscale(255 - binary)


# Read in this order; the first is also the image shown in the UI.
_PRIMARY_PREPROCESSORS = (
    _preprocess_crop_for_llm,
    _preprocess_saturated,
    _preprocess_binary_padded,
)
# Only used as a tie-breaker when the primary reads disagree.
_TIEBREAK_PREPROCESSOR = _preprocess_binary_inverted


def _ask_ollama(
    crop: np.ndarray,
    ollama_url: str,
    model: str,
    timeout: int = 60,
    processed: np.ndarray | None = None,
) -> tuple[str, float]:
    """Send crop to Ollama, return (raw_response, elapsed_seconds).

    Pass an already-preprocessed image as ``processed`` to send exactly those pixels.
    """
    if processed is None:
        processed = _preprocess_crop_for_llm(crop)
    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        raise RuntimeError("Could not encode crop for Ollama")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    payload = json.dumps({
        "model": model,
        "prompt": _PROMPT,
        "images": [image_b64],
        "stream": False,
        "keep_alive": -1,
        "think": False,
        "seed": random.randint(1, 2**31),
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    raw = result["response"]
    # Strip any <think>...</think> block the model emits despite think:False.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = _normalise_lookalike_ones(raw)
    # Extract the first run of 2-3 digits from the response — the model often
    # adds explanation text despite being told not to.
    m = re.search(r"\b(\d{2,3})\b", raw)
    if m:
        raw = m.group(1)
    return raw, elapsed


def _interpret(raw: str) -> Reading:
    """Turn a raw LLM digit string into a Reading using digit-count format rules."""
    cleaned = raw.replace(" ", "").replace(".", "")

    n = len(cleaned)

    if n == 3:
        upper = cleaned.upper()
        code = "FILL" if upper == "FIL" else upper if upper in KNOWN_ERROR_CODES else None
        if code is not None:
            return Reading(
                value=None,
                valid=False,
                raw_pattern=raw,
                confidence=1.0,
                display_text=cleaned,
                error_code=code,
                unit=None,
            )
        if cleaned.isdigit():
            value = float(f"{cleaned[:2]}.{cleaned[2]}")
            return Reading(
                value=value,
                valid=True,
                raw_pattern=raw,
                confidence=1.0,
                display_text=cleaned,
                error_code=None,
                unit="°C",
            )

    if n == 2 and cleaned.isdigit():
        value = float(f"{cleaned[0]}.{cleaned[1]}")
        return Reading(
            value=value,
            valid=True,
            raw_pattern=raw,
            confidence=1.0,
            display_text=cleaned,
            error_code=None,
            unit="bar",
        )

    return Reading(
        value=None,
        valid=False,
        raw_pattern=raw,
        confidence=0.0,
        display_text=cleaned or None,
        error_code=None,
        unit=None,
    )


def _read_once(
    crop: np.ndarray, ollama_url: str, model: str, processed: np.ndarray
) -> str | None:
    """One LLM read; None when the request itself failed."""
    try:
        raw, elapsed = _ask_ollama(crop, ollama_url, model, processed=processed)
    except Exception as exc:
        _log("ocr_request_failed", error_type=type(exc).__name__, error=str(exc))
        return None
    _log("ocr_llm_read", reply=raw, seconds=round(elapsed, 1))
    return raw


def _vote(reads: list[str]) -> tuple[str | None, int]:
    """Return (winning reply, votes) or (None, top votes) when the top two tie."""
    ranked = collections.Counter(reads).most_common(2)
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return ranked[0]
    return None, ranked[0][1]


def read_display(
    crop: np.ndarray,
    ollama_url: str = "",
    model: str = "lfm3-vl:latest",
    on_prepared: Callable[[np.ndarray], None] | None = None,
) -> tuple[Reading, np.ndarray]:
    """Read a located display crop via Ollama LLM and return (Reading, processed_image).

    The crop is read with each primary preprocessing.  If they all agree the
    reading has confidence 1.0.  Otherwise one more differently preprocessed
    read breaks the tie and the majority wins, with confidence = its share of
    the votes.  With no majority the Reading is invalid (value None) and
    raw_pattern lists the candidate replies separated by "/".  Failed requests
    are not votes; if every request fails LLMUnavailableError is raised.

    ollama_url defaults to the OLLAMA_URL env var if not supplied.
    crop should be a grayscale or BGR image of just the digit row.
    """
    if not ollama_url:
        ollama_url = os.getenv("OLLAMA_URL", "").strip()
    if not ollama_url:
        raise RuntimeError(
            "OLLAMA_URL is not configured. Set it in Settings or the OLLAMA_URL env var."
        )

    first = _PRIMARY_PREPROCESSORS[0](crop)
    if on_prepared is not None:
        # Called just before the request so the UI can show what is being sent.
        on_prepared(first)

    attempts = [
        _read_once(crop, ollama_url, model, first if i == 0 else prep(crop))
        for i, prep in enumerate(_PRIMARY_PREPROCESSORS)
    ]
    if len({a for a in attempts if a is not None}) > 1:
        attempts.append(_read_once(crop, ollama_url, model, _TIEBREAK_PREPROCESSOR(crop)))

    # A failed request is not a vote: only replies the model actually gave count.
    reads = [a for a in attempts if a is not None]
    if not reads:
        raise LLMUnavailableError(f"No reply from the LLM server at {ollama_url}")

    winner, votes = _vote(reads)
    if winner is None:
        candidates = "/".join(sorted(set(r for r in reads if r)))
        reading = Reading(
            value=None,
            valid=False,
            raw_pattern=candidates,
            confidence=0.0,
            display_text=candidates or None,
            error_code=None,
            unit=None,
        )
    else:
        reading = dataclasses.replace(_interpret(winner), confidence=votes / len(reads))
        if not reading.valid:
            reading = dataclasses.replace(reading, confidence=0.0)

    reading = dataclasses.replace(reading, llm_replies=tuple(reads))

    _log(
        "ocr_result",
        reads=reads,
        valid=reading.valid,
        value=reading.value,
        unit=reading.unit,
        confidence=round(reading.confidence, 2),
    )
    return reading, first
