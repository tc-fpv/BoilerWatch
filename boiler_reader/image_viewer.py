"""Serve the latest gauge frame with its region of interest highlighted."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Type
from urllib.parse import urlsplit

import cv2
import numpy as np
from dotenv import load_dotenv

from boiler_reader.decoder import DecoderConfig, DigitBox, ROI, Reading, decode_frame
from boiler_reader.locator import crop_display, locate_display, located_decoder_config
from boiler_reader.ocr import read_display
from boiler_reader.camera import CameraPoller
from boiler_reader.webhook import HomeAssistantWebhookNotifier, create_from_env
from boiler_reader.alert_delay import AlertDelayTracker, DelayStatus


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


# Keys written to / read from reader-settings.json.
_SETTINGS_KEYS = (
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
}


def _parse_box(value: str) -> DigitBox:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("boxes must use x,y,width,height coordinates")
    return DigitBox(*parts)


def _load_admin_settings(settings_state_path: Path) -> dict[str, object]:
    """Load persisted settings, seeding from env vars on first boot."""
    if settings_state_path.is_file():
        try:
            saved = json.loads(settings_state_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                # Fill any missing keys with defaults (handles upgrades).
                return {**_SETTINGS_DEFAULTS, **{k: saved[k] for k in _SETTINGS_KEYS if k in saved}}
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
    }
    _persist_json(settings_state_path, seeded)
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

    saved = _load_admin_settings(settings_state_path)

    poll_seconds = int(saved["poll_seconds"])
    low_threshold = float(saved["low_threshold"])
    high_threshold = float(saved["high_threshold"])
    alert_delay_minutes = int(saved["alert_delay_minutes"])
    _validate_reader_settings(poll_seconds, low_threshold, high_threshold, alert_delay_minutes)

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
    )


def _validate_reader_settings(
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


def _validate_admin_settings(s: dict[str, object]) -> None:
    _validate_reader_settings(
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


def _persist_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload), encoding="utf-8")
    temporary_path.replace(path)


def _camera_status(image_path: Path) -> dict[str, object]:
    status_path = image_path.with_name("camera-status.json")
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict):
                return status
        except (OSError, json.JSONDecodeError):
            pass
    return {"available": image_path.is_file(), "error": None}


def _send_due_webhook(
    notifier: HomeAssistantWebhookNotifier | None,
    delay_tracker: AlertDelayTracker,
    delay: DelayStatus,
    payload: dict[str, object],
    delivery_lock: Lock,
) -> None:
    if notifier is None or not delay.notification_due:
        return
    with delivery_lock:
        if delay_tracker.notification_due() and notifier.send(payload):
            delay_tracker.mark_notified()


def annotate_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    """Return a copy of an image with the ROI boundary highlighted."""
    height, width = image.shape[:2]
    if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI must have a non-negative origin and positive dimensions")
    if roi.x + roi.width > width or roi.y + roi.height > height:
        raise ValueError("ROI extends beyond the image")

    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        annotated = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        annotated = image.copy()
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA")

    thickness = max(1, round(min(width, height) / 300))
    cv2.rectangle(
        annotated,
        (roi.x, roi.y),
        (roi.x + roi.width - 1, roi.y + roi.height - 1),
        (0, 100, 255),
        thickness,
    )
    return annotated


def _decoder_config(settings: ViewerSettings, roi: ROI) -> DecoderConfig:
    return DecoderConfig(
        roi=roi,
        digit_boxes=settings.digit_boxes or (DigitBox(0, 0, roi.width, roi.height),),
        decimal_box=settings.decimal_box,
        expected_format=settings.expected_format,
    )


def _page(refresh_seconds: int) -> bytes:
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Boiler Reader</title>
<style>
:root{{color-scheme:dark;font-family:Consolas,'Courier New',monospace;background:#101514;color:#e8efeb}}
*{{box-sizing:border-box}}
html,body{{width:100%;height:100%;margin:0;overflow:hidden}}
body{{display:flex;background:#101514}}
main{{flex:1;min-width:0;display:flex;align-items:center;justify-content:center;background:#080b0a;padding:16px}}
#stage{{position:relative;display:inline-block;max-width:100%;max-height:100%}}
#frame{{display:block;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain}}
#roi-box{{position:absolute;border:2px solid #ff7a35;cursor:move;touch-action:none;min-width:8px;min-height:8px}}
#resize-handle{{position:absolute;right:-7px;bottom:-7px;width:14px;height:14px;padding:0;border:2px solid #101514;background:#ff7a35;cursor:nwse-resize;touch-action:none}}
aside{{flex:0 0 310px;padding:28px 24px;background:#17201c;border-left:1px solid #35433a;display:flex;flex-direction:column;gap:22px}}
h1{{margin:0;color:#9bad9f;font-size:12px;font-weight:600;letter-spacing:1px}}
.reading{{display:flex;align-items:baseline;gap:10px;min-height:66px}}
#value{{font-size:52px;line-height:1;color:#f1f6f2}}
#unit{{font-size:15px;color:#9bad9f}}
#status{{align-self:flex-start;border:1px solid #668875;color:#bde2c7;padding:6px 9px;font-size:12px}}
#status.invalid{{border-color:#b77568;color:#f0b2a4}}
dl{{margin:0;display:grid;grid-template-columns:1fr;gap:7px}}
dt{{margin-top:9px;color:#829187;font-size:11px;text-transform:uppercase}}
dd{{margin:0;overflow-wrap:anywhere;font-size:14px;color:#e0e9e3}}
#pattern{{font-size:12px;color:#bac8be}}
time{{margin-top:auto;color:#718077;font-size:11px}}
form{{display:grid;gap:10px;border-top:1px solid #35433a;padding-top:16px}}
label{{display:grid;grid-template-columns:1fr 90px;align-items:center;gap:8px;color:#9bad9f;font-size:12px}}
input{{width:90px;padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit}}
button.save{{justify-self:start;padding:8px 12px;border:1px solid #829987;background:#26372e;color:#edf4ee;font:inherit;cursor:pointer}}
button.save:disabled{{opacity:.6;cursor:wait}}
.actions{{display:grid;gap:8px;border-top:1px solid #35433a;padding-top:14px}}
button.action{{width:100%;padding:8px 10px;border:1px solid #526258;background:#202c25;color:#edf4ee;font:inherit;text-align:left;cursor:pointer}}
button.action:disabled{{opacity:.6;cursor:wait}}
#last-grabbed{{margin-top:0;color:#9bad9f}}
#admin-overlay:not([hidden]){{display:flex}}
@media(max-width:700px){{body{{flex-direction:column}}main{{min-height:0}}aside{{flex:0 0 auto;width:100%;padding:14px 18px;gap:10px;border-left:0;border-top:1px solid #35433a}}.reading{{min-height:48px}}#value{{font-size:38px}}dl{{grid-template-columns:1fr 1fr}}dt{{margin-top:3px}}time{{margin-top:4px}}}}
</style></head>
<body>
<div id="admin-overlay" hidden style="position:fixed;inset:0;background:rgba(0,0,0,.7);align-items:center;justify-content:center;z-index:100">
<div style="background:#17201c;border:1px solid #35433a;padding:28px 24px;width:min(480px,95vw);max-height:90vh;overflow-y:auto;display:flex;flex-direction:column;gap:16px">
<div style="display:flex;justify-content:space-between;align-items:center"><h2 style="margin:0;color:#9bad9f;font-size:13px;font-weight:600;letter-spacing:1px">SETTINGS</h2><button id="admin-close" type="button" style="background:none;border:none;color:#829187;font-size:18px;cursor:pointer;padding:0 4px">&#x2715;</button></div>
<form id="admin-form" style="display:grid;gap:10px">
<p style="margin:0;font-size:11px;color:#829187">Changes take effect immediately. RTSP settings restart the camera connection.</p>
<fieldset style="border:1px solid #35433a;padding:12px 14px;margin:0"><legend style="color:#829187;font-size:11px;padding:0 6px">CAMERA</legend>
<div style="display:grid;gap:8px">
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">RTSP URL<input id="a-rtsp-url" type="text" placeholder="rtsp://host/stream" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Username<input id="a-rtsp-user" type="text" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Password<input id="a-rtsp-pass" type="password" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Poll interval (s)<input id="a-poll-seconds" type="number" min="1" max="3600" step="1" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
</div></fieldset>
<fieldset style="border:1px solid #35433a;padding:12px 14px;margin:0"><legend style="color:#829187;font-size:11px;padding:0 6px">HOME ASSISTANT</legend>
<div style="display:grid;gap:8px">
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Webhook URL<input id="a-ha-webhook-url" type="password" placeholder="http://ha-host:8123/api/webhook/..." style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Timeout (s)<input id="a-ha-webhook-timeout" type="number" min="1" max="60" step="1" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
</div></fieldset>
<fieldset style="border:1px solid #35433a;padding:12px 14px;margin:0"><legend style="color:#829187;font-size:11px;padding:0 6px">ALERTS</legend>
<div style="display:grid;gap:8px">
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Low limit (bar)<input id="a-low-threshold" type="number" min="0" max="10" step="0.1" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">High limit (bar)<input id="a-high-threshold" type="number" min="0" max="10" step="0.1" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Alert delay (min)<input id="a-alert-delay-minutes" type="number" min="1" max="1440" step="1" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
</div></fieldset>
<fieldset style="border:1px solid #35433a;padding:12px 14px;margin:0"><legend style="color:#829187;font-size:11px;padding:0 6px">DETECTION</legend>
<div style="display:grid;gap:8px">
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Brightness threshold<input id="a-brightness-threshold" type="number" min="0" max="255" step="1" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"></label>
<label style="display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:8px;color:#9bad9f;font-size:12px">Expected format<select id="a-expected-format" style="padding:7px;border:1px solid #526258;background:#0f1512;color:#f1f6f2;font:inherit;width:100%"><option value="auto">auto</option><option value="d.d">d.d (pressure)</option><option value="dd.d">dd.d (temperature)</option></select></label>
</div></fieldset>
<button id="admin-save" type="submit" style="justify-self:start;padding:8px 16px;border:1px solid #829987;background:#26372e;color:#edf4ee;font:inherit;cursor:pointer">Save</button>
</form></div></div>
<main><div id="stage"><img id="frame" src="/last.jpg" alt="Latest camera frame with ROI"><div id="roi-box" title="Drag to move ROI"><button id="resize-handle" type="button" title="Drag to resize ROI" aria-label="Resize ROI"></button></div></div></main>
<aside><h1>GAUGE DECODER</h1><div class="reading"><span id="value">--</span><span id="unit">bar</span></div>
<div id="status">WAITING</div><dl><dt>Display</dt><dd id="display">--</dd>
<dt>Error code</dt><dd id="error-code">--</dd>
<dt>Issue timer</dt><dd id="issue-delay">No active issue</dd>
<dt>Segment pattern</dt><dd id="pattern">--</dd><dt>Confidence</dt><dd id="confidence">--</dd></dl>
<div class="actions"><button class="action" id="capture-now" type="button">Grab new frame</button>
<button class="action" id="send-test-webhook" type="button">Send test HA message</button>
<button class="action" id="admin-btn" type="button">Settings</button>
<time id="last-grabbed">Last frame: unknown</time></div>
<time id="updated">Waiting for reading</time></aside>
<script>
let refreshMs={refresh_seconds * 1000};
const frame=document.getElementById('frame');
const stage=document.getElementById('stage');
const roiBox=document.getElementById('roi-box');
let imageSize=null;
let currentRoi=null;
let drag=null;
let saving=false;
function drawRoi(){{
    if(!currentRoi||!imageSize||!frame.naturalWidth)return;
    const imageRect=frame.getBoundingClientRect();
    const stageRect=stage.getBoundingClientRect();
    const scaleX=imageRect.width/imageSize.width;
    const scaleY=imageRect.height/imageSize.height;
    roiBox.style.left=(imageRect.left-stageRect.left+currentRoi.x*scaleX)+'px';
    roiBox.style.top=(imageRect.top-stageRect.top+currentRoi.y*scaleY)+'px';
    roiBox.style.width=(currentRoi.width*scaleX)+'px';
    roiBox.style.height=(currentRoi.height*scaleY)+'px';
}}
function showReading(reading){{
    imageSize=reading.image_size||imageSize;
    currentRoi=reading.roi||currentRoi;
    const valid=reading.valid;
    const unit=reading.unit||'bar';
    document.getElementById('value').textContent=valid?Number(reading.value).toFixed(1):'--';
    document.getElementById('unit').textContent=unit;
    document.getElementById('unit').hidden=!valid;
    document.getElementById('display').textContent=reading.display_text||'--';
    document.getElementById('error-code').textContent=reading.error_code||'--';
    const delay=reading.delay;
    document.getElementById('issue-delay').textContent=delay&&delay.issue
        ?(delay.issue==='low_pressure'?'Low pressure: ':'Error: ')+(delay.eligible?'ready':delay.remaining_seconds+'s remaining')
        :'No active issue';
    document.getElementById('pattern').textContent=reading.raw_pattern||'--';
    document.getElementById('confidence').textContent=Math.round(reading.confidence*100)+'%';
    const status=document.getElementById('status');
    status.textContent=valid?(unit==='°C'?'VALID TEMPERATURE':'VALID PRESSURE'):'INVALID / ERROR';
    status.classList.toggle('invalid',!valid);
    document.getElementById('updated').textContent='Scanned '+new Date().toLocaleTimeString();
    if(reading.camera)showLastGrabbed(reading.camera);
    drawRoi();
}}
function showLastGrabbed(camera){{
    const lastGrabbed=document.getElementById('last-grabbed');
    if(!camera||!camera.checked_at){{lastGrabbed.textContent='Last frame: unknown';return;}}
    const timestamp=new Date(camera.checked_at).toLocaleString();
    lastGrabbed.textContent=(camera.available?'Last frame: ':'Last grab failed: ')+timestamp;
}}
async function refresh(){{
        if(drag||saving)return;
        frame.src='/last.jpg?t='+Date.now();
    const status=document.getElementById('status');
    try{{
        const response=await fetch('/reading.json?t='+Date.now(),{{cache:'no-store'}});
        const reading=await response.json();
        if(!response.ok)throw new Error(reading.error||'Reading unavailable');
        showReading(reading);
    }}catch(error){{
        document.getElementById('value').textContent='--';
        document.getElementById('unit').hidden=true;
        document.getElementById('display').textContent=error.message;
        document.getElementById('error-code').textContent='--';
        document.getElementById('issue-delay').textContent=error.message;
        document.getElementById('pattern').textContent='--';
        document.getElementById('confidence').textContent='--';
        status.textContent='NO IMAGE';
        status.classList.add('invalid');
        document.getElementById('updated').textContent='Waiting for latest frame';
        fetch('/camera-status.json?t='+Date.now(),{{cache:'no-store'}})
            .then(response=>response.json()).then(showLastGrabbed).catch(()=>{{}});
    }}
}}
frame.addEventListener('load',()=>{{frame.hidden=false;drawRoi();}});
frame.addEventListener('error',()=>{{frame.hidden=true;}});
document.getElementById('capture-now').addEventListener('click',async event=>{{
    const button=event.currentTarget;
    button.disabled=true;button.textContent='Grabbing frame...';
    try{{
        const response=await fetch('/capture',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Camera grab failed');
        showLastGrabbed(result.camera);
        await refresh();
        document.getElementById('updated').textContent='Frame grabbed and decoded';
    }}catch(error){{
        document.getElementById('status').textContent='CAMERA GRAB FAILED';
        document.getElementById('status').classList.add('invalid');
        document.getElementById('updated').textContent=error.message;
    }}finally{{button.disabled=false;button.textContent='Grab new frame';}}
}});
document.getElementById('send-test-webhook').addEventListener('click',async event=>{{
    if(!confirm('Send a simulated ECS error to Home Assistant? This may trigger a mobile notification.'))return;
    const button=event.currentTarget;
    button.disabled=true;button.textContent='Sending test...';
    try{{
        const response=await fetch('/test-webhook',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Webhook delivery failed');
        document.getElementById('updated').textContent='Test HA message sent';
    }}catch(error){{
        document.getElementById('updated').textContent=error.message;
    }}finally{{button.disabled=false;button.textContent='Send test HA message';}}
}});
roiBox.addEventListener('pointerdown',event=>{{
    if(!imageSize||!currentRoi)return;
    event.preventDefault();
    const resizing=event.target.id==='resize-handle';
    drag={{pointerId:event.pointerId,resizing,startX:event.clientX,startY:event.clientY,roi:{{...currentRoi}}}};
    roiBox.setPointerCapture(event.pointerId);
}});
roiBox.addEventListener('pointermove',event=>{{
    if(!drag||drag.pointerId!==event.pointerId)return;
    const rect=frame.getBoundingClientRect();
    const dx=Math.round((event.clientX-drag.startX)*imageSize.width/rect.width);
    const dy=Math.round((event.clientY-drag.startY)*imageSize.height/rect.height);
    if(drag.resizing){{
        currentRoi.width=Math.max(25,Math.min(imageSize.width-drag.roi.x,drag.roi.width+dx));
        currentRoi.height=Math.max(25,Math.min(imageSize.height-drag.roi.y,drag.roi.height+dy));
    }}else{{
        currentRoi.x=Math.max(0,Math.min(imageSize.width-drag.roi.width,drag.roi.x+dx));
        currentRoi.y=Math.max(0,Math.min(imageSize.height-drag.roi.height,drag.roi.y+dy));
    }}
    drawRoi();
}});
async function commitRoi(){{
    if(!drag)return;
    drag=null;saving=true;
    const status=document.getElementById('status');
    status.textContent='SCANNING';
    try{{
        const response=await fetch('/roi',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(currentRoi)}});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Could not update ROI');
        showReading(result);
    }}catch(error){{
        status.textContent='ROI UPDATE FAILED';
        status.classList.add('invalid');
        document.getElementById('updated').textContent=error.message;
        refresh();
    }}finally{{saving=false;}}
}}
roiBox.addEventListener('pointerup',commitRoi);
roiBox.addEventListener('pointercancel',commitRoi);
window.addEventListener('resize',drawRoi);
let refreshTimer=setInterval(refresh,refreshMs);
document.getElementById('admin-btn').addEventListener('click',async()=>{{
    const adminOverlay=document.getElementById('admin-overlay');
    adminOverlay.hidden=false;
    adminOverlay.style.display='flex';
    try{{
        const r=await fetch('/admin-settings');
        const s=await r.json();
        document.getElementById('a-rtsp-url').value=s.rtsp_url||'';
        document.getElementById('a-rtsp-user').value=s.rtsp_user||'';
        document.getElementById('a-rtsp-pass').value=s.rtsp_pass||'';
        document.getElementById('a-poll-seconds').value=s.poll_seconds||30;
        document.getElementById('a-ha-webhook-url').value=s.ha_webhook_url||'';
        document.getElementById('a-ha-webhook-timeout').value=s.ha_webhook_timeout||5;
        document.getElementById('a-low-threshold').value=s.low_threshold||1.0;
        document.getElementById('a-high-threshold').value=s.high_threshold||1.5;
        document.getElementById('a-alert-delay-minutes').value=s.alert_delay_minutes||5;
        document.getElementById('a-brightness-threshold').value=s.brightness_threshold||30;
        document.getElementById('a-expected-format').value=s.expected_format||'auto';
    }}catch(e){{
        document.getElementById('updated').textContent='Could not load admin settings: '+e.message;
    }}
}});
function closeAdmin(){{const o=document.getElementById('admin-overlay');o.hidden=true;o.style.display='none';}}
document.addEventListener('click',e=>{{const o=document.getElementById('admin-overlay');if(o&&e.target===o)closeAdmin();}});
document.addEventListener('keydown',e=>{{const o=document.getElementById('admin-overlay');if(e.key==='Escape'&&o&&!o.hidden)closeAdmin();}});
document.addEventListener('click',e=>{{if(e.target&&e.target.id==='admin-close')closeAdmin();}});
document.getElementById('admin-form').addEventListener('submit',async event=>{{
    event.preventDefault();
    const button=document.getElementById('admin-save');
    button.disabled=true;button.textContent='Saving...';
    try{{
        const payload={{
            rtsp_url:document.getElementById('a-rtsp-url').value.trim(),
            rtsp_user:document.getElementById('a-rtsp-user').value.trim(),
            rtsp_pass:document.getElementById('a-rtsp-pass').value,
            poll_seconds:Number(document.getElementById('a-poll-seconds').value),
            ha_webhook_url:document.getElementById('a-ha-webhook-url').value.trim(),
            ha_webhook_timeout:Number(document.getElementById('a-ha-webhook-timeout').value),
            low_threshold:Number(document.getElementById('a-low-threshold').value),
            high_threshold:Number(document.getElementById('a-high-threshold').value),
            alert_delay_minutes:Number(document.getElementById('a-alert-delay-minutes').value),
            brightness_threshold:Number(document.getElementById('a-brightness-threshold').value),
            expected_format:document.getElementById('a-expected-format').value,
        }};
        const response=await fetch('/admin-settings',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Could not save admin settings');
        closeAdmin();
        document.getElementById('updated').textContent='Admin settings saved';
    }}catch(error){{
        document.getElementById('updated').textContent=error.message;
    }}finally{{button.disabled=false;button.textContent='Save';}}
}});
refresh();
</script>
</body></html>"""
    return html.encode("utf-8")


def make_handler(
    settings: ViewerSettings,
    camera_poller: CameraPoller | None = None,
    delay_tracker: AlertDelayTracker | None = None,
    reader_settings: dict[str, object] | None = None,
    settings_lock: RLock | None = None,
    roi_state: list[ROI] | None = None,
    webhook_notifier: HomeAssistantWebhookNotifier | None = None,
    webhook_delivery_lock: Lock | None = None,
    cached_reading_state: list[dict[str, object] | None] | None = None,
    cached_reading_state_lock: RLock | None = None,
    webhook_notifier_state: list[HomeAssistantWebhookNotifier | None] | None = None,
    on_admin_save: object = None,
) -> Type[BaseHTTPRequestHandler]:
    current_roi: list[ROI | None] = roi_state if roi_state is not None else [settings.roi]
    roi_lock = RLock()
    cached_reading: list[dict[str, object] | None] = cached_reading_state if cached_reading_state is not None else [None]
    cached_reading_lock: RLock = cached_reading_state_lock if cached_reading_state_lock is not None else RLock()
    delay_tracker = delay_tracker or AlertDelayTracker(
        settings.alert_delay_minutes * 60
    )
    current_settings = reader_settings if reader_settings is not None else {
        "poll_seconds": settings.poll_seconds,
        "low_threshold": settings.low_threshold,
        "high_threshold": settings.high_threshold,
        "alert_delay_minutes": settings.alert_delay_minutes,
    }
    settings_lock = settings_lock or RLock()
    webhook_delivery_lock = webhook_delivery_lock or Lock()
    # Use the shared notifier list if provided (allows live updates from admin panel).
    _notifier_ref: list[HomeAssistantWebhookNotifier | None] = (
        webhook_notifier_state if webhook_notifier_state is not None
        else [webhook_notifier]
    )

    def get_roi() -> ROI | None:
        with roi_lock:
            return current_roi[0]

    def format_reading(reading: Reading, image: np.ndarray, roi: ROI) -> dict[str, object]:
        reading_fields = asdict(reading)
        with settings_lock:
            low_threshold = float(current_settings["low_threshold"])
            high_threshold = float(current_settings["high_threshold"])
        if not reading_fields["valid"]:
            issue = "error"
        elif reading_fields["unit"] == "°C":
            issue = None
        elif float(reading_fields["value"]) < low_threshold:
            issue = "low_pressure"
        elif float(reading_fields["value"]) > high_threshold:
            issue = "error"
        else:
            issue = None
        delay_status = delay_tracker.update(issue)
        status = (
            "low" if issue == "low_pressure"
            else "invalid" if issue == "error"
            else "ok"
        )
        with settings_lock:
            delay_minutes = int(current_settings["alert_delay_minutes"])
        _send_due_webhook(
            _notifier_ref[0],
            delay_tracker,
            delay_status,
            {
                "status": status,
                "issue": issue,
                "pressure": reading_fields["value"] if reading_fields["unit"] == "bar" else None,
                "temperature": reading_fields["value"] if reading_fields["unit"] == "°C" else None,
                "display": reading_fields["display_text"],
                "error_code": reading_fields["error_code"],
                "delay_minutes": delay_minutes,
            },
            webhook_delivery_lock,
        )
        return {
            **reading_fields,
            "roi": asdict(roi),
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "camera": _camera_status(settings.image_path),
            "issue": issue,
            "delay": asdict(delay_status),
        }

    def decode_latest(image: np.ndarray) -> dict[str, object]:
        roi = get_roi()
        if roi is None or (settings.auto_locate and settings.digit_boxes is None):
            with settings_lock:
                bt = int(current_settings.get("brightness_threshold", settings.brightness_threshold))
            located = locate_display(image, bt)
            if located is None:
                result: dict[str, object] = {
                    "valid": False,
                    "value": None,
                    "unit": None,
                    "raw_pattern": "",
                    "confidence": 0.0,
                    "display_text": None,
                    "error_code": None,
                    "roi": None,
                    "image_size": {"width": image.shape[1], "height": image.shape[0]},
                    "camera": _camera_status(settings.image_path),
                    "issue": "error",
                    "delay": {"issue": "error", "remaining_seconds": 0, "eligible": False, "notification_due": False},
                }
                with cached_reading_lock:
                    cached_reading[0] = result
                return result
            effective_roi = located.roi
            reading, _ = read_display(crop_display(image, located))
        else:
            effective_roi = roi
            reading = decode_frame(image, _decoder_config(settings, effective_roi))
        result = format_reading(reading, image, effective_roi)
        with cached_reading_lock:
            cached_reading[0] = result
        return result

    def get_cached_reading() -> dict[str, object] | None:
        with cached_reading_lock:
            return cached_reading[0]

    def get_reader_settings() -> dict[str, object]:
        with settings_lock:
            return dict(current_settings)

    class ImageHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_path = urlsplit(self.path).path
            if request_path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return

            if request_path == "/":
                body = _page(settings.refresh_seconds)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if request_path == "/reading.json":
                cached = get_cached_reading()
                if cached is not None:
                    # Re-evaluate the delay tracker and webhook on each poll
                    # so the issue timer advances and alerts fire on schedule.
                    issue = cached.get("issue")
                    delay_status = delay_tracker.update(issue)
                    with settings_lock:
                        delay_minutes = int(current_settings["alert_delay_minutes"])
                    status = (
                        "low" if issue == "low_pressure"
                        else "invalid" if issue == "error"
                        else "ok"
                    )
                    _send_due_webhook(
                        _notifier_ref[0],
                        delay_tracker,
                        delay_status,
                        {
                            "status": status,
                            "issue": issue,
                            "pressure": cached.get("value") if cached.get("unit") == "bar" else None,
                            "temperature": cached.get("value") if cached.get("unit") == "°C" else None,
                            "display": cached.get("display_text"),
                            "error_code": cached.get("error_code"),
                            "delay_minutes": delay_minutes,
                        },
                        webhook_delivery_lock,
                    )
                    cached = {
                        **cached,
                        "camera": _camera_status(settings.image_path),
                        "delay": asdict(delay_status),
                    }
                    self._send_json(200, cached)
                    return
                # No cache yet — decode once on first request.
                image = cv2.imread(str(settings.image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    camera_status = _camera_status(settings.image_path)
                    delay = delay_tracker.update("error")
                    with settings_lock:
                        delay_minutes = int(current_settings["alert_delay_minutes"])
                    _send_due_webhook(
                        _notifier_ref[0],
                        delay_tracker,
                        delay,
                        {
                            "status": "stale",
                            "issue": "error",
                            "pressure": None,
                            "error": camera_status.get("error") or "Camera frame unavailable",
                            "delay_minutes": delay_minutes,
                        },
                        webhook_delivery_lock,
                    )
                    self._send_json(
                        503,
                        {
                            "error": camera_status.get("error") or "Latest image is not available",
                            "camera": camera_status,
                            "issue": "error",
                            "delay": asdict(delay_tracker.update("error")),
                        },
                    )
                    return
                try:
                    reading = decode_latest(image)
                except ValueError as error:
                    self._send_json(500, {"error": str(error)})
                    return
                self._send_json(200, reading)
                return

            if request_path == "/settings.json":
                self._send_json(200, get_reader_settings())
                return

            if request_path == "/admin-settings":
                with settings_lock:
                    payload = {k: current_settings[k] for k in _SETTINGS_KEYS if k in current_settings}
                self._send_json(200, payload)
                return

            if request_path == "/camera-status.json":
                self._send_json(200, _camera_status(settings.image_path))
                return

            if request_path != "/last.jpg":
                self.send_error(404)
                return

            image = cv2.imread(str(settings.image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.send_error(503, "Camera has not supplied a frame")
                return
            roi_for_annotation = get_roi()
            if roi_for_annotation is None:
                located = locate_display(image, settings.brightness_threshold)
                roi_for_annotation = located.roi if located is not None else None
            try:
                if roi_for_annotation is not None:
                    annotated = annotate_roi(image, roi_for_annotation)
                else:
                    if image.ndim == 2:
                        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif image.ndim == 3 and image.shape[2] == 4:
                        annotated = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    else:
                        annotated = image.copy()
            except ValueError as error:
                self.send_error(500, str(error))
                return
            ok, encoded = cv2.imencode(".jpg", annotated)
            if not ok:
                self.send_error(500, "Could not encode latest image")
                return
            body = encoded.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request_path = urlsplit(self.path).path
            if request_path == "/capture":
                if camera_poller is None:
                    self._send_json(503, {"error": "RTSP_URL is not configured"})
                    return
                captured = camera_poller.capture_once()
                camera = _camera_status(settings.image_path)
                if captured:
                    cached = get_cached_reading()
                    if cached is not None:
                        self._send_json(200, {**cached, "captured": True, "camera": camera})
                        return
                self._send_json(
                    200 if captured else 502,
                    {
                        "captured": captured,
                        "camera": camera,
                        "error": camera.get("error"),
                    },
                )
                return
            if request_path == "/test-webhook":
                if _notifier_ref[0] is None:
                    self._send_json(503, {"error": "HA_WEBHOOK_URL is not configured"})
                    return
                delivered = _notifier_ref[0].send(
                    {
                        "status": "invalid",
                        "issue": "error",
                        "pressure": None,
                        "display": "ECS",
                        "error_code": "ECS",
                        "error": "Manual test message",
                        "delay_minutes": 0,
                        "simulated": True,
                    }
                )
                self._send_json(
                    200 if delivered else 502,
                    {"delivered": delivered, "error": None if delivered else "Webhook delivery failed"},
                )
                return
            if request_path == "/settings":
                self._update_reader_settings()
                return
            if request_path == "/admin-settings":
                self._update_admin_settings()
                return
            if request_path != "/roi":
                self.send_error(404)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 1024:
                    raise ValueError("ROI request body must be between 1 and 1024 bytes")
                payload = json.loads(self.rfile.read(content_length))
                roi = ROI(
                    x=int(payload["x"]),
                    y=int(payload["y"]),
                    width=int(payload["width"]),
                    height=int(payload["height"]),
                )
                image = cv2.imread(str(settings.image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    self._send_json(503, {"error": "Latest image is not available"})
                    return
                annotate_roi(image, roi)
                if settings.digit_boxes is not None:
                    config = _decoder_config(settings, roi)
                else:
                    located = locate_display(image, settings.brightness_threshold)
                    if located is not None:
                        from boiler_reader.locator import located_decoder_config as _ldc
                        config = _ldc(
                            located.__class__(
                                roi=roi,
                                digit_boxes=located.digit_boxes,
                                decimal_box=located.decimal_box,
                                confidence=located.confidence,
                            ),
                            settings.expected_format,
                        )
                    else:
                        config = _decoder_config(settings, roi)
                reading = decode_frame(image, config)
                if settings.roi_state_path is not None:
                    state_path = settings.roi_state_path
                    _persist_json(state_path, asdict(roi))
                with roi_lock:
                    current_roi[0] = roi
                self._send_json(
                    200,
                    format_reading(reading, image, roi),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})
            except OSError as error:
                self._send_json(500, {"error": f"Could not save ROI: {error}"})

        def _update_reader_settings(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 1024:
                    raise ValueError("Settings request body must be between 1 and 1024 bytes")
                payload = json.loads(self.rfile.read(content_length))
                poll_seconds_raw = float(payload["poll_seconds"])
                if not math.isfinite(poll_seconds_raw) or not poll_seconds_raw.is_integer():
                    raise ValueError("POLL_SECONDS must be a whole number")
                poll_seconds = int(poll_seconds_raw)
                low_threshold = float(payload["low_threshold"])
                high_threshold = float(payload["high_threshold"])
                alert_delay_raw = float(payload["alert_delay_minutes"])
                if not math.isfinite(alert_delay_raw) or not alert_delay_raw.is_integer():
                    raise ValueError("ALERT_DELAY_MINUTES must be a whole number")
                alert_delay_minutes = int(alert_delay_raw)
                _validate_reader_settings(
                    poll_seconds,
                    low_threshold,
                    high_threshold,
                    alert_delay_minutes,
                )
                partial = {
                    "poll_seconds": poll_seconds,
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "alert_delay_minutes": alert_delay_minutes,
                }
                with settings_lock:
                    current_settings.update(partial)
                    full = {k: current_settings[k] for k in _SETTINGS_KEYS if k in current_settings}
                if settings.settings_state_path is not None:
                    _persist_json(settings.settings_state_path, full)
                with cached_reading_lock:
                    cached_reading[0] = None
                if camera_poller is not None:
                    camera_poller.set_poll_seconds(poll_seconds)
                delay_tracker.set_delay_seconds(alert_delay_minutes * 60)
                self._send_json(200, partial)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})
            except OSError as error:
                self._send_json(500, {"error": f"Could not save settings: {error}"})

        def _update_admin_settings(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 4096:
                    raise ValueError("Admin settings body must be between 1 and 4096 bytes")
                payload = json.loads(self.rfile.read(content_length))
                saved: dict[str, object] = {
                    "rtsp_url": str(payload.get("rtsp_url", "")).strip(),
                    "rtsp_user": str(payload.get("rtsp_user", "")).strip(),
                    "rtsp_pass": str(payload.get("rtsp_pass", "")),
                    "poll_seconds": int(float(payload.get("poll_seconds", 30))),
                    "ha_webhook_url": str(payload.get("ha_webhook_url", "")).strip(),
                    "ha_webhook_timeout": int(payload.get("ha_webhook_timeout", 5)),
                    "low_threshold": float(payload.get("low_threshold", 1.0)),
                    "high_threshold": float(payload.get("high_threshold", 1.5)),
                    "alert_delay_minutes": int(float(payload.get("alert_delay_minutes", 5))),
                    "brightness_threshold": int(payload.get("brightness_threshold", 30)),
                    "expected_format": str(payload.get("expected_format", "auto")),
                }
                _validate_admin_settings(saved)
                with settings_lock:
                    current_settings.update(saved)
                    full = {k: current_settings[k] for k in _SETTINGS_KEYS if k in current_settings}
                if settings.settings_state_path is not None:
                    _persist_json(settings.settings_state_path, full)
                with cached_reading_lock:
                    cached_reading[0] = None
                # Update webhook notifier.
                if webhook_notifier_state is not None:
                    webhook_notifier_state[0] = _make_notifier(saved)
                # Update delay tracker and camera poller with new values.
                delay_tracker.set_delay_seconds(int(saved["alert_delay_minutes"]) * 60)
                if camera_poller is not None:
                    camera_poller.set_poll_seconds(int(saved["poll_seconds"]))
                # Restart camera poller if RTSP settings changed.
                if on_admin_save is not None:
                    rtsp_url = saved["rtsp_url"]
                    if rtsp_url:
                        on_admin_save(
                            str(rtsp_url),
                            str(saved["rtsp_user"]),
                            str(saved["rtsp_pass"]),
                            int(saved["poll_seconds"]),
                        )
                self._send_json(200, {"ok": True})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})
            except OSError as error:
                self._send_json(500, {"error": f"Could not save admin settings: {error}"})

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"image-viewer: {format % args}")

    return ImageHandler


def _make_notifier(s: dict[str, object]) -> HomeAssistantWebhookNotifier | None:
    url = str(s.get("ha_webhook_url", "")).strip()
    if not url:
        return None
    return HomeAssistantWebhookNotifier(
        url,
        timeout_seconds=int(s.get("ha_webhook_timeout", 5)),
    )


def serve(settings: ViewerSettings | None = None) -> None:
    settings = settings or settings_from_env()
    camera_poller: CameraPoller | None = None
    camera_thread: Thread | None = None
    camera_stop_event = Event()
    camera_lock = Lock()
    stop_event = Event()
    delay_tracker = AlertDelayTracker(settings.alert_delay_minutes * 60)
    webhook_notifier: list[HomeAssistantWebhookNotifier | None] = [
        _make_notifier({
            "ha_webhook_url": settings.ha_webhook_url,
            "ha_webhook_timeout": settings.ha_webhook_timeout,
        })
    ]
    webhook_delivery_lock = Lock()
    active_roi: list[ROI | None] = [settings.roi]
    shared_cached_reading: list[dict[str, object] | None] = [None]
    shared_cached_reading_lock = RLock()
    reader_settings: dict[str, object] = {
        "poll_seconds": settings.poll_seconds,
        "low_threshold": settings.low_threshold,
        "high_threshold": settings.high_threshold,
        "alert_delay_minutes": settings.alert_delay_minutes,
        "brightness_threshold": settings.brightness_threshold,
        "expected_format": settings.expected_format,
        "rtsp_url": settings.rtsp_url,
        "rtsp_user": settings.rtsp_user,
        "rtsp_pass": settings.rtsp_pass,
        "ha_webhook_url": settings.ha_webhook_url,
        "ha_webhook_timeout": settings.ha_webhook_timeout,
    }
    settings_lock = RLock()

    def send_for_issue(issue: str | None, payload: dict[str, object]) -> None:
        delay = delay_tracker.update(issue)
        with settings_lock:
            payload["delay_minutes"] = int(reader_settings["alert_delay_minutes"])
        _send_due_webhook(
            webhook_notifier[0],
            delay_tracker,
            delay,
            payload,
            webhook_delivery_lock,
        )

    def save_error_frame(frame: np.ndarray, label: str, processed: np.ndarray | None = None) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        folder = settings.image_path.parent / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{ts}_{label}.jpg"), frame)
        if processed is not None:
            cv2.imwrite(str(folder / f"{ts}_{label}_ocr.jpg"), processed)

    last_reading_state: list[tuple[float | None, str | None]] = [None, None]

    def observe_frame(frame: np.ndarray | None, error: str | None) -> None:
        if frame is None or error is not None:
            send_for_issue(
                "error",
                {
                    "status": "stale",
                    "issue": "error",
                    "pressure": None,
                    "error": error or "Camera frame unavailable",
                },
            )
            with shared_cached_reading_lock:
                shared_cached_reading[0] = None
            return
        # Decode the new frame and cache the result.
        ocr_processed: np.ndarray | None = None
        try:
            roi = active_roi[0]
            if roi is None or (settings.auto_locate and settings.digit_boxes is None):
                with settings_lock:
                    bt = int(reader_settings["brightness_threshold"])
                located = locate_display(frame, bt)
                if located is None:
                    save_error_frame(frame, "ERROR")
                    send_for_issue(
                        "error",
                        {
                            "status": "invalid",
                            "issue": "error",
                            "pressure": None,
                            "error": "Display could not be located in frame",
                        },
                    )
                    with shared_cached_reading_lock:
                        shared_cached_reading[0] = None
                    return
                reading, ocr_processed = read_display(crop_display(frame, located))
                effective_roi = located.roi
            else:
                reading = decode_frame(frame, _decoder_config(settings, roi))
                effective_roi = roi
        except ValueError:
            save_error_frame(frame, "ERROR")
            send_for_issue(
                "error",
                {
                    "status": "invalid",
                    "issue": "error",
                    "pressure": None,
                    "error": "Gauge frame could not be decoded",
                },
            )
            with shared_cached_reading_lock:
                shared_cached_reading[0] = None
            return
        with settings_lock:
            low_threshold = float(reader_settings["low_threshold"])
            high_threshold = float(reader_settings["high_threshold"])
        if not reading.valid:
            issue = "error"
            save_error_frame(frame, "ERROR", ocr_processed)
        elif reading.unit == "°C":
            issue = None
        elif reading.value is not None and reading.value < low_threshold:
            issue = "low_pressure"
            save_error_frame(frame, "LOW", ocr_processed)
        elif reading.value is not None and reading.value > high_threshold:
            issue = "error"
            save_error_frame(frame, "HIGH", ocr_processed)
        else:
            issue = None
        # Detect suspicious valid readings: low confidence or implausible jump in value.
        if reading.valid and reading.unit == "bar" and reading.value is not None:
            confidence_too_low = reading.confidence < 0.5
            last_value, last_unit = last_reading_state
            value_jump_too_large = (
                last_value is not None
                and last_unit == "bar"
                and abs(reading.value - last_value) > 1.5
            )
            if confidence_too_low or value_jump_too_large:
                reason = "LOW_CONF" if confidence_too_low else "JUMP"
                save_error_frame(frame, reason, ocr_processed)
        # Update last reading for next frame's delta comparison.
        if reading.valid:
            last_reading_state[0] = reading.value
            last_reading_state[1] = reading.unit
        send_for_issue(
            issue,
            {
                "status": (
                    "low" if issue == "low_pressure"
                    else "invalid" if issue == "error"
                    else "ok"
                ),
                "issue": issue,
                "pressure": reading.value if reading.unit == "bar" else None,
                "temperature": reading.value if reading.unit == "°C" else None,
                "display": reading.display_text,
                "error_code": reading.error_code,
            },
        )
        cached = {
            **asdict(reading),
            "roi": asdict(effective_roi),
            "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
            "camera": _camera_status(settings.image_path),
            "issue": issue,
            "delay": asdict(delay_tracker.update(issue)),
        }
        with shared_cached_reading_lock:
            shared_cached_reading[0] = cached

    def start_camera(rtsp_url: str, rtsp_user: str, rtsp_pass: str, poll_secs: int) -> None:
        nonlocal camera_poller, camera_thread, camera_stop_event
        with camera_lock:
            # Stop existing poller if running.
            if camera_poller is not None:
                camera_stop_event.set()
                if camera_thread is not None:
                    camera_thread.join(timeout=2)
            camera_stop_event = Event()
            camera_poller = CameraPoller(
                rtsp_url=rtsp_url,
                username=rtsp_user,
                password=rtsp_pass,
                image_path=settings.image_path,
                poll_seconds=poll_secs,
                on_frame=observe_frame,
            )
            camera_poller.capture_once()
            camera_thread = Thread(
                target=camera_poller.run,
                kwargs={"stop_event": camera_stop_event, "capture_on_start": False},
                daemon=True,
            )
            camera_thread.start()

    if settings.rtsp_url:
        start_camera(
            settings.rtsp_url,
            settings.rtsp_user,
            settings.rtsp_pass,
            settings.poll_seconds,
        )

    server = ThreadingHTTPServer(
        ("0.0.0.0", settings.port),
        make_handler(
            settings,
            camera_poller,
            delay_tracker,
            reader_settings,
            settings_lock,
            active_roi,
            webhook_notifier[0],
            webhook_delivery_lock,
            shared_cached_reading,
            shared_cached_reading_lock,
            webhook_notifier_state=webhook_notifier,
            on_admin_save=start_camera,
        ),
    )
    print(f"image-viewer: serving on port {settings.port}; image={settings.image_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        camera_stop_event.set()
        server.server_close()
        if camera_thread is not None:
            camera_thread.join(timeout=2)


def main() -> None:
    serve()


if __name__ == "__main__":
    main()