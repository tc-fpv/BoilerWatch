"""LLM-based reading of the located seven-segment display crop.

Sends a preprocessed crop to a local Ollama VLM (default: minicpm-v4.6:latest)
and interprets the returned digits using digit-count format rules:
  2 digits → pressure  (e.g. "12" → 1.2 bar)
  3 digits → temperature or error code  (e.g. "321" → 32.1 °C, "ECS" → error)
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time
import urllib.request

import cv2
import numpy as np

from boiler_reader.decoder import KNOWN_ERROR_CODES, Reading


_UPSCALE = 4

_PROMPT = (
    "This is a cropped, upscaled seven-segment display from a boiler gauge. "
    "The digits are bright white on a dark background. "
    "Read the digits shown. "
    "Reply with only the digits, e.g. '12' or '321'. "
    "Do not include decimal points or units. Do not explain or add any other text."
)


def _preprocess_crop_for_llm(crop: np.ndarray) -> np.ndarray:
    """Grayscale + 4x upscale, bright segments on dark background."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    return cv2.resize(gray, None, fx=_UPSCALE, fy=_UPSCALE, interpolation=cv2.INTER_CUBIC)


def _ask_ollama(crop: np.ndarray, ollama_url: str, model: str, timeout: int = 60) -> tuple[str, float]:
    """Send crop to Ollama, return (raw_response, elapsed_seconds)."""
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


def read_display(
    crop: np.ndarray,
    ollama_url: str = "",
    model: str = "minicpm-v4.6:latest",
) -> tuple[Reading, np.ndarray]:
    """Read a located display crop via Ollama LLM and return (Reading, processed_image).

    ollama_url defaults to the OLLAMA_URL env var if not supplied.
    crop should be a grayscale or BGR image of just the digit row.
    """
    if not ollama_url:
        ollama_url = os.getenv("OLLAMA_URL", "").strip()
    if not ollama_url:
        raise RuntimeError(
            "OLLAMA_URL is not configured. Set it in Settings or the OLLAMA_URL env var."
        )

    processed = _preprocess_crop_for_llm(crop)

    try:
        raw, elapsed = _ask_ollama(crop, ollama_url, model)
    except Exception as exc:
        print(f"ocr: ollama request failed: {exc}")
        invalid = Reading(
            value=None,
            valid=False,
            raw_pattern="",
            confidence=0.0,
            display_text=None,
            error_code=None,
            unit=None,
        )
        return invalid, processed

    reading = _interpret(raw)
    print(
        f"ocr: llm={raw!r} -> "
        f"{'valid' if reading.valid else 'invalid'} "
        f"{reading.value} {reading.unit or ''} "
        f"({elapsed:.1f}s)"
    )
    return reading, processed
