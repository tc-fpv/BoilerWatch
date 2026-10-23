"""Standalone test: push fixture images through a local Ollama VLM and compare readings.

Run with:
    python tests/test_ollama_ocr.py [--host http://192.168.0.26:11434]
    python tests/test_ollama_ocr.py --model qwen2.5vl   # single model, all fixtures
    python tests/test_ollama_ocr.py --benchmark          # all models, first fixture only

Requires Ollama running with a vision model pulled.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

_PROMPT = (
    "This is a cropped, upscaled seven-segment display from a boiler gauge. "
    "The digits are bright white on a dark background. "
    "Read the digits shown. "
    "Reply with only the digits, e.g. '12' or '321'. "
    "Do not include decimal points or units. Do not explain or add any other text."
)

_ALL_MODELS = ["qwen2.5vl:3b", "minicpm-v4.6:latest", "qwen2.5vl:latest"]
_BENCHMARK_MODELS = _ALL_MODELS

_PROCESSED_DIR = Path(__file__).with_name("fixtures") / "processed"
_FIXTURES_DIR  = Path(__file__).with_name("fixtures")
_SCREENSHOTS_DIR = Path(__file__).with_name("fixtures")


def _load_fixtures() -> list[tuple[Path, str]]:
    """Load all images from the processed folder, parsing expected value from filename."""
    import re
    fixtures = []
    for path in sorted(_PROCESSED_DIR.glob("*.jpg")) if _PROCESSED_DIR.exists() else []:
        m = re.search(r"([0-9.]+)\s*(bar|C)(?:_|\.|$)", path.name, re.IGNORECASE)
        if m:
            value, unit = m.group(1), m.group(2).upper()
            fixtures.append((path, f"{value} {unit}"))
        else:
            fixtures.append((path, None))
    return fixtures


def _parse_expected(name: str) -> str | None:
    """Extract expected value from filename suffix e.g. '..._ocr_60.2C.jpg' -> '60.2 C'."""
    import re
    m = re.search(r"_ocr_([0-9.]+)(bar|C)\.jpg$", name, re.IGNORECASE)
    if not m:
        return None
    value, unit = m.group(1), m.group(2).upper()
    return f"{value} {unit}"


def _preprocess(image_path: Path) -> bytes:
    import cv2
    from boiler_reader.locator import crop_display, locate_display
    from boiler_reader.ocr import _preprocess_crop
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    located = locate_display(image)
    if located is None:
        raise RuntimeError(f"Could not locate display in {image_path.name}")
    crop = crop_display(image, located)
    processed = _preprocess_crop(crop)
    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        raise RuntimeError(f"Could not encode preprocessed image for {image_path.name}")
    return encoded.tobytes()


def _ask_ollama(host: str, model: str, image_bytes: bytes) -> tuple[str, float]:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
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
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return result["response"].strip(), elapsed


def run_benchmark(host: str) -> None:
    image_path = _FIXTURES_DIR / "boiler-pressure.jpg"
    expected = "1.2 bar"
    print(f"Benchmark image: {image_path.name}  (expected: {expected})\n")
    image_bytes = _preprocess(image_path)
    for model in _BENCHMARK_MODELS:
        print(f"  [{model}] ...", end="", flush=True)
        try:
            result, elapsed = _ask_ollama(host, model, image_bytes)
            ok = expected.lower() in result.lower()
            status = "PASS" if ok else "FAIL"
            print(f"\r[{status}] {model:<20} got={result!r:<16} {elapsed:.2f}s")
        except Exception as exc:
            print(f"\r[ERR ] {model:<20} {str(exc)[:80]}")


def run_all_fixtures(host: str, model: str) -> None:
    print(f"Model: {model}  Host: {host}\n")
    fixtures = _load_fixtures()
    if not fixtures:
        print(f"No images found in {_PROCESSED_DIR}")
        return
    passed = 0
    failed = 0
    skipped = 0
    for image_path, expected in fixtures:
        print(f"  testing {image_path.name}...", end="", flush=True)
        image_bytes = image_path.read_bytes()
        result, elapsed = _ask_ollama(host, model, image_bytes)
        if expected is None:
            print(f"\r[SKIP] {image_path.name:<40} got={result!r:<16} {elapsed:.2f}s")
            skipped += 1
            continue
        expected_digits = expected.split()[0].replace(".", "")
        ok = expected_digits in result.replace(".", "").replace(" ", "")
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\r[{status}] {image_path.name:<40} expected={expected!r:<12} got={result!r:<16} {elapsed:.2f}s")
    print(f"\nModel: {model}  —  {passed}/{passed + failed} passed" + (f", {skipped} skipped" if skipped else ""))


def run_screenshots(host: str, model: str) -> None:
    print(f"Model: {model}  Host: {host}\n")
    ocr_files = sorted(_SCREENSHOTS_DIR.glob("*_ocr_*.jpg"))
    if not ocr_files:
        print(f"No _ocr_ files found in {_SCREENSHOTS_DIR}")
        return
    passed = 0
    failed = 0
    skipped = 0
    for image_path in ocr_files:
        expected = _parse_expected(image_path.name)
        if expected is None:
            print(f"[SKIP] {image_path.name} (no expected value in filename)")
            skipped += 1
            continue
        print(f"  testing {image_path.name}...", end="", flush=True)
        image_bytes = image_path.read_bytes()
        result, elapsed = _ask_ollama(host, model, image_bytes)
        ok = expected.lower().replace(" ", "") in result.lower().replace(" ", "")
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\r[{status}] {image_path.name:<45} expected={expected!r:<12} got={result!r:<16} {elapsed:.2f}s")
    print(f"\nModel: {model}  —  {passed}/{passed + failed} passed" + (f", {skipped} skipped" if skipped else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://192.168.0.26:11434")
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmark", action="store_true", help="test all models against one image")
    parser.add_argument("--screenshots", action="store_true", help="test _ocr_ files in tests/fixtures/")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.host)
    elif args.screenshots:
        models = [args.model] if args.model else _ALL_MODELS
        for model in models:
            run_screenshots(args.host, model)
            print()
    else:
        models = [args.model] if args.model else _ALL_MODELS
        for model in models:
            run_all_fixtures(args.host, model)
            print()


if __name__ == "__main__":
    main()
    sys.exit(0)
