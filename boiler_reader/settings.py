"""Startup configuration and persisted, UI-editable reader settings."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from boiler_reader.decoder import DigitBox, ROI


@dataclass(frozen=True)
class ViewerSettings:
    image_path: Path
    roi: ROI | None = None
    port: int = 8080
    refresh_seconds: int = 2
    digit_boxes: tuple[DigitBox, ...] | None = None
    decimal_box: DigitBox | None = None
    expected_format: str = "auto"
    roi_state_path: Path | None = None
    poll_seconds: int = 30
    low_threshold: float = 1.0
    high_threshold: float = 1.5
    settings_state_path: Path | None = None
    alert_delay_minutes: int = 5
    auto_locate: bool = True
    brightness_threshold: int = 30
    rtsp_url: str = ""
    rtsp_user: str = ""
    rtsp_pass: str = ""
    ha_webhook_url: str = ""
    ha_webhook_timeout: int = 5
    ollama_url: str = ""
    ollama_model: str = "lfm3-vl:latest"


# Keys written to / read from reader-settings.json.
SETTINGS_KEYS = (
    "poll_seconds",
    "low_threshold",
    "high_threshold",
    "alert_delay_minutes",
    "brightness_threshold",
    "expected_format",
    "rtsp_url",
    "rtsp_user",
    "rtsp_pass",
    "ha_webhook_url",
    "ha_webhook_timeout",
    "ollama_url",
    "ollama_model",
)

_SETTINGS_DEFAULTS: dict[str, object] = {
    "poll_seconds": 30,
    "low_threshold": 1.0,
    "high_threshold": 1.5,
    "alert_delay_minutes": 5,
    "brightness_threshold": 30,
    "expected_format": "auto",
    "rtsp_url": "",
    "rtsp_user": "",
    "rtsp_pass": "",
    "ha_webhook_url": "",
    "ha_webhook_timeout": 5,
    "ollama_url": "",
    "ollama_model": "lfm3-vl:latest",
}


def persist_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload), encoding="utf-8")
    temporary_path.replace(path)


def validate_reader_settings(
    poll_seconds: int,
    low_threshold: float,
    high_threshold: float,
    alert_delay_minutes: int,
) -> None:
    if not 1 <= poll_seconds <= 3600:
        raise ValueError("poll_seconds must be between 1 and 3600")
    if not math.isfinite(low_threshold) or not 0 <= low_threshold <= 10:
        raise ValueError("low_threshold must be between 0 and 10 bar")
    if not math.isfinite(high_threshold) or not 0 <= high_threshold <= 10:
        raise ValueError("high_threshold must be between 0 and 10 bar")
    if high_threshold <= low_threshold:
        raise ValueError("high_threshold must be greater than low_threshold")
    if not 1 <= alert_delay_minutes <= 1440:
        raise ValueError("alert_delay_minutes must be between 1 and 1440")


def validate_admin_settings(s: dict[str, object]) -> None:
    validate_reader_settings(
        int(s["poll_seconds"]),
        float(s["low_threshold"]),
        float(s["high_threshold"]),
        int(s["alert_delay_minutes"]),
    )
    if not 0 <= int(s["brightness_threshold"]) <= 255:
        raise ValueError("brightness_threshold must be between 0 and 255")
    if str(s["expected_format"]) not in {"auto", "d", "d.d", "dd.d"}:
        raise ValueError("expected_format must be auto, d, d.d, or dd.d")
    if not 1 <= int(s["ha_webhook_timeout"]) <= 60:
        raise ValueError("ha_webhook_timeout must be between 1 and 60")
    ollama_url = str(s.get("ollama_url", "")).strip()
    if ollama_url and not (ollama_url.startswith("http://") or ollama_url.startswith("https://")):
        raise ValueError("ollama_url must start with http:// or https://")


def load_admin_settings(settings_state_path: Path) -> dict[str, object]:
    """Load persisted settings, seeding from env vars on first boot."""
    if settings_state_path.is_file():
        try:
            saved = json.loads(settings_state_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                # Fill any missing keys with defaults (handles upgrades).
                return {**_SETTINGS_DEFAULTS, **{k: saved[k] for k in SETTINGS_KEYS if k in saved}}
        except (OSError, json.JSONDecodeError):
            pass

    # First boot — seed from env vars, write to file so UI can manage them.
    seeded: dict[str, object] = {
        **_SETTINGS_DEFAULTS,
        "rtsp_url": os.getenv("RTSP_URL", "").strip(),
        "rtsp_user": os.getenv("RTSP_USER", "").strip(),
        "rtsp_pass": os.getenv("RTSP_PASS", "").strip(),
        "ha_webhook_url": os.getenv("HA_WEBHOOK_URL", "").strip(),
        "ha_webhook_timeout": int(os.getenv("HA_WEBHOOK_TIMEOUT", "5")),
        "poll_seconds": int(os.getenv("POLL_SECONDS", "30")),
        "low_threshold": float(os.getenv("LOW_THRESHOLD", os.getenv("LOWER_LIMIT", "1.0"))),
        "high_threshold": float(os.getenv("HIGH_THRESHOLD", "1.5")),
        "alert_delay_minutes": int(os.getenv("ALERT_DELAY_MINUTES", "5")),
        "brightness_threshold": int(os.getenv("BRIGHTNESS_THRESHOLD", "30")),
        "expected_format": os.getenv("EXPECTED_FORMAT", "auto"),
        "ollama_url": os.getenv("OLLAMA_URL", "").strip(),
        "ollama_model": os.getenv("OLLAMA_MODEL", "lfm3-vl:latest").strip(),
    }
    persist_json(settings_state_path, seeded)
    return seeded


def settings_from_env() -> ViewerSettings:
    """Load viewer settings, seeding from env vars on first boot only."""
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    port = int(os.getenv("WEB_PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("WEB_PORT must be between 1 and 65535")

    image_path = Path("/data/last.jpg")
    roi_state_path = Path("/data/roi.json")
    settings_state_path = Path("/data/reader-settings.json")

    saved = load_admin_settings(settings_state_path)

    poll_seconds = int(saved["poll_seconds"])
    low_threshold = float(saved["low_threshold"])
    high_threshold = float(saved["high_threshold"])
    alert_delay_minutes = int(saved["alert_delay_minutes"])
    validate_reader_settings(poll_seconds, low_threshold, high_threshold, alert_delay_minutes)

    roi: ROI | None = None
    if roi_state_path.is_file():
        try:
            saved_roi = json.loads(roi_state_path.read_text(encoding="utf-8"))
            roi = ROI(**saved_roi)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    return ViewerSettings(
        image_path=image_path,
        roi=roi,
        port=port,
        expected_format=str(saved["expected_format"]),
        roi_state_path=roi_state_path,
        poll_seconds=poll_seconds,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        settings_state_path=settings_state_path,
        alert_delay_minutes=alert_delay_minutes,
        brightness_threshold=int(saved["brightness_threshold"]),
        rtsp_url=str(saved["rtsp_url"]),
        rtsp_user=str(saved["rtsp_user"]),
        rtsp_pass=str(saved["rtsp_pass"]),
        ha_webhook_url=str(saved["ha_webhook_url"]),
        ha_webhook_timeout=int(saved["ha_webhook_timeout"]),
        ollama_url=str(saved.get("ollama_url", "")),
        ollama_model=str(saved.get("ollama_model", "lfm3-vl:latest")),
    )
