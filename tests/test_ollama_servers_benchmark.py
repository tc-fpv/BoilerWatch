"""Benchmark: fire the fixture images through both Ollama servers and record timings.

Compares OLLAMA_URL (production) against TEST_OLLAMA_URL (alternative), both read
from .env. Each image goes through the real pipeline (locate -> crop -> ocr._ask_ollama),
so the timings match what the app sees.

Opt-in because it is slow and needs both servers up:
    $env:RUN_LLM_BENCHMARK = "1"; pytest tests/test_ollama_servers_benchmark.py -s

Optional env vars:
    BENCHMARK_RUNS   timed runs per image per server (default 3)
    OLLAMA_MODEL     model used on both servers (default from .env)

There is no warm-up: both servers keep the model loaded (keep_alive -1). A summary table is printed and also written to
tests/fixtures/processed/ollama_benchmark.json.
"""

from __future__ import annotations

import json
import os
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pytest
from dotenv import load_dotenv

from boiler_reader.locator import crop_display, locate_display
from boiler_reader.ocr import _ask_ollama, _interpret

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=False)

_FIXTURES = Path(__file__).with_name("fixtures")
_RESULTS_FILE = _FIXTURES / "processed" / "ollama_benchmark.json"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Expected readings (value) for fixtures whose filename does not carry one.
_EXPECTED = {
    "boiler-pressure.jpg": 1.2,
    "boiler-temperature.jpg": 52.3,
    "boiler-temperature-cover-angle.png": 34.3,
}

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_BENCHMARK") != "1",
    reason="set RUN_LLM_BENCHMARK=1 to run the Ollama server benchmark",
)


def _expected_value(name: str) -> float | None:
    import re

    if name in _EXPECTED:
        return _EXPECTED[name]
    m = re.search(r"([0-9]+\.[0-9])\s*(?:bar|C)", name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _servers() -> dict[str, str]:
    servers = {
        "OLLAMA_URL": os.getenv("OLLAMA_URL", "").strip(),
        "TEST_OLLAMA_URL": os.getenv("TEST_OLLAMA_URL", "").strip(),
    }
    missing = [k for k, v in servers.items() if not v]
    if missing:
        pytest.skip(f"not set in .env / environment: {', '.join(missing)}")
    return servers


def _require_reachable(label: str, url: str) -> None:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=5):
            pass
    except Exception as exc:
        pytest.fail(f"{label} ({url}) is not reachable: {exc}")


def _fixture_crops() -> list[tuple[str, object]]:
    crops = []
    for path in sorted(_FIXTURES.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        located = locate_display(image)
        if located is None:
            print(f"skipping {path.name}: display not located")
            continue
        crops.append((path.name, crop_display(image, located)))
    return crops


def test_compare_ollama_server_processing_times() -> None:
    servers = _servers()
    model = os.getenv("OLLAMA_MODEL", "minicpm-v4.6:latest").strip()
    runs = int(os.getenv("BENCHMARK_RUNS", "3"))
    for label, url in servers.items():
        _require_reachable(label, url)

    crops = _fixture_crops()
    assert crops, f"no locatable images found in {_FIXTURES}"

    print(
        f"\n{len(crops)} images x {len(servers)} servers x {runs} runs "
        f"= {len(crops) * len(servers) * runs} timed requests (model {model})",
        flush=True,
    )

    results: dict[str, dict[str, dict]] = {}
    for idx, (name, crop) in enumerate(crops, 1):
        expected = _expected_value(name)
        results[name] = {}
        print(f"\n[{idx}/{len(crops)}] {name}  (expected {expected})", flush=True)
        for label, url in servers.items():
            times: list[float] = []
            readings: list[str] = []
            print(f"  {label:<16}", end="", flush=True)
            for _ in range(runs):
                raw, elapsed = _ask_ollama(crop, url, model, timeout=120)
                times.append(elapsed)
                readings.append(raw)
                print(f" {raw!r} {elapsed:.2f}s", end="", flush=True)
            print(f"  -> mean {statistics.mean(times):.2f}s", flush=True)
            values = [_interpret(r).value for r in readings]
            correct = (
                None if expected is None
                else sum(v is not None and abs(v - expected) < 0.05 for v in values)
            )
            results[name][label] = {
                "mean_s": statistics.mean(times),
                "min_s": min(times),
                "max_s": max(times),
                "times_s": times,
                "readings": readings,
                "expected": expected,
                "correct": correct,
                "runs": runs,
            }

    labels = list(servers)
    print(f"\nModel: {model}   runs/image: {runs}   (times are mean seconds per request)")
    for label, url in servers.items():
        print(f"  {label:<16} {url}")
    header = f"\n{'image':<44}" + "".join(f"{lbl:>20}" for lbl in labels)
    print(header)
    print("-" * len(header.strip("\n")))
    for name, per_server in results.items():
        cells = ""
        for label in labels:
            r = per_server[label]
            acc = "" if r["correct"] is None else f" {r['correct']}/{r['runs']}ok"
            cells += f"{r['mean_s']:>13.2f}s{acc:>6}"
        print(f"{name:<44}{cells}")
    overall = {
        label: statistics.mean(r[label]["mean_s"] for r in results.values()) for label in labels
    }
    print("-" * len(header.strip("\n")))
    print(f"{'overall mean':<44}" + "".join(f"{overall[lbl]:>19.2f}s" for lbl in labels))

    _RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_FILE.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "servers": servers,
                "overall_mean_s": overall,
                "images": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {_RESULTS_FILE}")

    for per_server in results.values():
        for r in per_server.values():
            assert all(t > 0 for t in r["times_s"])
