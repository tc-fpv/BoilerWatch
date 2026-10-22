"""Serve the latest gauge frame with its region of interest highlighted."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Type
from urllib.parse import urlsplit

import cv2
import numpy as np
from dotenv import load_dotenv

from boiler_reader.decoder import DecoderConfig, DigitBox, ROI, Reading, decode_frame
from boiler_reader.camera import CameraPoller
from boiler_reader.webhook import HomeAssistantWebhookNotifier, create_from_env
from boiler_reader.alert_delay import AlertDelayTracker, DelayStatus


@dataclass(frozen=True)
class ViewerSettings:
    image_path: Path
    roi: ROI
    port: int = 8080
    refresh_seconds: int = 2
    digit_boxes: tuple[DigitBox, ...] = (
        DigitBox(0, 0, 22, 85),
        DigitBox(30, 3, 48, 76),
    )
    decimal_box: DigitBox | None = DigitBox(13, 68, 9, 8)
    expected_format: str = "auto"
    roi_state_path: Path | None = None
    poll_seconds: int = 30
    low_threshold: float = 1.0
    settings_state_path: Path | None = None
    alert_delay_minutes: int = 5


def _parse_box(value: str) -> DigitBox:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("boxes must use x,y,width,height coordinates")
    return DigitBox(*parts)


def settings_from_env() -> ViewerSettings:
    """Read viewer settings, using the current Reolink frame as ROI defaults."""
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    port = int(os.getenv("WEB_PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("WEB_PORT must be between 1 and 65535")
    digit_boxes = tuple(
        _parse_box(value)
        for value in os.getenv("DIGIT_BOXES", "0,0,22,85;30,3,48,76").split(";")
        if value.strip()
    )
    decimal_value = os.getenv("DECIMAL_BOX", "13,68,9,8").strip()
    image_path = Path(os.getenv("LAST_IMAGE_PATH", "/data/last.jpg"))
    roi = ROI(
        x=int(os.getenv("ROI_X", "330")),
        y=int(os.getenv("ROI_Y", "238")),
        width=int(os.getenv("ROI_W", "110")),
        height=int(os.getenv("ROI_H", "94")),
    )
    roi_state_path = Path(
        os.getenv("ROI_STATE_PATH", str(image_path.with_name("roi.json")))
    )
    settings_state_path = Path(
        os.getenv("READER_SETTINGS_PATH", str(image_path.with_name("reader-settings.json")))
    )
    poll_seconds = int(os.getenv("POLL_SECONDS", "30"))
    low_threshold = float(
        os.getenv("LOW_THRESHOLD", os.getenv("LOWER_LIMIT", "1.0"))
    )
    alert_delay_minutes = int(os.getenv("ALERT_DELAY_MINUTES", "5"))
    if roi_state_path.is_file():
        saved_roi = json.loads(roi_state_path.read_text(encoding="utf-8"))
        roi = ROI(**saved_roi)
    if settings_state_path.is_file():
        saved_settings = json.loads(settings_state_path.read_text(encoding="utf-8"))
        poll_seconds = int(saved_settings.get("poll_seconds", poll_seconds))
        low_threshold = float(saved_settings.get("low_threshold", low_threshold))
        alert_delay_minutes = int(
            saved_settings.get("alert_delay_minutes", alert_delay_minutes)
        )
    _validate_reader_settings(poll_seconds, low_threshold, alert_delay_minutes)
    return ViewerSettings(
        image_path=image_path,
        roi=roi,
        port=port,
        digit_boxes=digit_boxes,
        decimal_box=_parse_box(decimal_value) if decimal_value else None,
        expected_format=os.getenv("EXPECTED_FORMAT", "auto"),
        roi_state_path=roi_state_path,
        poll_seconds=poll_seconds,
        low_threshold=low_threshold,
        settings_state_path=settings_state_path,
        alert_delay_minutes=alert_delay_minutes,
    )


def _validate_reader_settings(
    poll_seconds: int,
    low_threshold: float,
    alert_delay_minutes: int,
) -> None:
    if not 1 <= poll_seconds <= 3600:
        raise ValueError("POLL_SECONDS must be between 1 and 3600")
    if not math.isfinite(low_threshold) or not 0 <= low_threshold <= 10:
        raise ValueError("LOW_THRESHOLD must be between 0 and 10 bar")
    if not 1 <= alert_delay_minutes <= 1440:
        raise ValueError("ALERT_DELAY_MINUTES must be between 1 and 1440")


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
        digit_boxes=settings.digit_boxes,
        decimal_box=settings.decimal_box,
        expected_format=settings.expected_format,
    )


def _page(
    refresh_seconds: int,
    poll_seconds: int,
    low_threshold: float,
    alert_delay_minutes: int,
) -> bytes:
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Boiler gauge</title>
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
@media(max-width:700px){{body{{flex-direction:column}}main{{min-height:0}}aside{{flex:0 0 auto;width:100%;padding:14px 18px;gap:10px;border-left:0;border-top:1px solid #35433a}}.reading{{min-height:48px}}#value{{font-size:38px}}dl{{grid-template-columns:1fr 1fr}}dt{{margin-top:3px}}time{{margin-top:4px}}}}
</style></head>
<body><main><div id="stage"><img id="frame" src="/last.jpg" alt="Latest camera frame with ROI"><div id="roi-box" title="Drag to move ROI"><button id="resize-handle" type="button" title="Drag to resize ROI" aria-label="Resize ROI"></button></div></div></main>
<aside><h1>GAUGE DECODER</h1><div class="reading"><span id="value">--</span><span id="unit">bar</span></div>
<div id="status">WAITING</div><dl><dt>Display</dt><dd id="display">--</dd>
<dt>Error code</dt><dd id="error-code">--</dd>
<dt>Issue timer</dt><dd id="issue-delay">No active issue</dd>
<dt>Segment pattern</dt><dd id="pattern">--</dd><dt>Confidence</dt><dd id="confidence">--</dd></dl>
<div class="actions"><button class="action" id="capture-now" type="button">Grab new frame</button>
<button class="action" id="send-test-webhook" type="button">Send test HA message</button>
<time id="last-grabbed">Last frame: unknown</time></div>
<form id="settings-form"><label for="poll-seconds">Poll seconds<input id="poll-seconds" type="number" min="1" max="3600" step="1" value="{poll_seconds}" required></label>
<label for="low-threshold">Low limit (bar)<input id="low-threshold" type="number" min="0" max="10" step="0.1" value="{low_threshold}" required></label>
<label for="alert-delay-minutes">Error delay (minutes)<input id="alert-delay-minutes" type="number" min="1" max="1440" step="1" value="{alert_delay_minutes}" required></label>
<button class="save" id="save-settings" type="submit">Save settings</button></form>
<time id="updated">Waiting for reading</time></aside>
<script>
let refreshMs={poll_seconds * 1000};
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
    document.getElementById('value').textContent=valid?Number(reading.value).toFixed(1):'--';
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
    status.textContent=valid?'VALID PRESSURE':'INVALID / ERROR';
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
document.getElementById('settings-form').addEventListener('submit',async event=>{{
    event.preventDefault();
    const button=document.getElementById('save-settings');
    button.disabled=true;
    try{{
        const response=await fetch('/settings',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{poll_seconds:Number(document.getElementById('poll-seconds').value),low_threshold:Number(document.getElementById('low-threshold').value),alert_delay_minutes:Number(document.getElementById('alert-delay-minutes').value)}})}});
        const result=await response.json();
        if(!response.ok)throw new Error(result.error||'Could not save settings');
        refreshMs=result.poll_seconds*1000;
        clearInterval(refreshTimer);
        refreshTimer=setInterval(refresh,refreshMs);
        document.getElementById('updated').textContent='Settings saved';
    }}catch(error){{
        document.getElementById('updated').textContent=error.message;
    }}finally{{button.disabled=false;}}
}});
refresh();
</script></body></html>"""
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
) -> Type[BaseHTTPRequestHandler]:
    current_roi = roi_state if roi_state is not None else [settings.roi]
    roi_lock = RLock()
    delay_tracker = delay_tracker or AlertDelayTracker(
        settings.alert_delay_minutes * 60
    )
    current_settings = reader_settings if reader_settings is not None else {
        "poll_seconds": settings.poll_seconds,
        "low_threshold": settings.low_threshold,
        "alert_delay_minutes": settings.alert_delay_minutes,
    }
    settings_lock = settings_lock or RLock()
    webhook_delivery_lock = webhook_delivery_lock or Lock()

    def get_roi() -> ROI:
        with roi_lock:
            return current_roi[0]

    def format_reading(reading: Reading, image: np.ndarray, roi: ROI) -> dict[str, object]:
        reading_fields = asdict(reading)
        with settings_lock:
            low_threshold = float(current_settings["low_threshold"])
        if not reading_fields["valid"]:
            issue = "error"
        elif float(reading_fields["value"]) < low_threshold:
            issue = "low_pressure"
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
            webhook_notifier,
            delay_tracker,
            delay_status,
            {
                "status": status,
                "issue": issue,
                "pressure": reading_fields["value"],
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
        reading = decode_frame(image, _decoder_config(settings, roi))
        return format_reading(reading, image, roi)

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
                current = get_reader_settings()
                body = _page(
                    settings.refresh_seconds,
                    int(current["poll_seconds"]),
                    float(current["low_threshold"]),
                    int(current["alert_delay_minutes"]),
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if request_path == "/reading.json":
                image = cv2.imread(str(settings.image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    camera_status = _camera_status(settings.image_path)
                    delay = delay_tracker.update("error")
                    with settings_lock:
                        delay_minutes = int(current_settings["alert_delay_minutes"])
                    _send_due_webhook(
                        webhook_notifier,
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
            try:
                annotated = annotate_roi(image, get_roi())
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
                if webhook_notifier is None:
                    self._send_json(503, {"error": "HA_WEBHOOK_URL is not configured"})
                    return
                delivered = webhook_notifier.send(
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
                reading = decode_frame(image, _decoder_config(settings, roi))
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
                alert_delay_raw = float(payload["alert_delay_minutes"])
                if not math.isfinite(alert_delay_raw) or not alert_delay_raw.is_integer():
                    raise ValueError("ALERT_DELAY_MINUTES must be a whole number")
                alert_delay_minutes = int(alert_delay_raw)
                _validate_reader_settings(
                    poll_seconds,
                    low_threshold,
                    alert_delay_minutes,
                )
                saved = {
                    "poll_seconds": poll_seconds,
                    "low_threshold": low_threshold,
                    "alert_delay_minutes": alert_delay_minutes,
                }
                if settings.settings_state_path is not None:
                    _persist_json(settings.settings_state_path, saved)
                with settings_lock:
                    current_settings.update(saved)
                if camera_poller is not None:
                    camera_poller.set_poll_seconds(poll_seconds)
                delay_tracker.set_delay_seconds(alert_delay_minutes * 60)
                self._send_json(200, saved)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})
            except OSError as error:
                self._send_json(500, {"error": f"Could not save settings: {error}"})

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


def serve(settings: ViewerSettings | None = None) -> None:
    settings = settings or settings_from_env()
    rtsp_url = os.getenv("RTSP_URL", "").strip()
    camera_poller = None
    camera_thread = None
    stop_event = Event()
    delay_tracker = AlertDelayTracker(settings.alert_delay_minutes * 60)
    webhook_notifier = create_from_env()
    webhook_delivery_lock = Lock()
    active_roi = [settings.roi]
    reader_settings: dict[str, object] = {
        "poll_seconds": settings.poll_seconds,
        "low_threshold": settings.low_threshold,
        "alert_delay_minutes": settings.alert_delay_minutes,
    }
    settings_lock = RLock()

    def send_for_issue(issue: str | None, payload: dict[str, object]) -> None:
        delay = delay_tracker.update(issue)
        with settings_lock:
            payload["delay_minutes"] = int(reader_settings["alert_delay_minutes"])
        _send_due_webhook(
            webhook_notifier,
            delay_tracker,
            delay,
            payload,
            webhook_delivery_lock,
        )

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
            return
        try:
            reading = decode_frame(frame, _decoder_config(settings, active_roi[0]))
        except ValueError:
            send_for_issue(
                "error",
                {
                    "status": "invalid",
                    "issue": "error",
                    "pressure": None,
                    "error": "Gauge frame could not be decoded",
                },
            )
            return
        with settings_lock:
            low_threshold = float(reader_settings["low_threshold"])
        if not reading.valid:
            issue = "error"
        elif reading.value is not None and reading.value < low_threshold:
            issue = "low_pressure"
        else:
            issue = None
        send_for_issue(
            issue,
            {
                "status": (
                    "low" if issue == "low_pressure"
                    else "invalid" if issue == "error"
                    else "ok"
                ),
                "issue": issue,
                "pressure": reading.value,
                "display": reading.display_text,
                "error_code": reading.error_code,
            },
        )

    if rtsp_url:
        camera_poller = CameraPoller(
            rtsp_url=rtsp_url,
            username=os.getenv("RTSP_USER", ""),
            password=os.getenv("RTSP_PASS", ""),
            image_path=settings.image_path,
            poll_seconds=settings.poll_seconds,
            on_frame=observe_frame,
        )
        camera_poller.capture_once()
        camera_thread = Thread(
            target=camera_poller.run,
            kwargs={"stop_event": stop_event, "capture_on_start": False},
            daemon=True,
        )
        camera_thread.start()

    server = ThreadingHTTPServer(
        ("0.0.0.0", settings.port),
        make_handler(
            settings,
            camera_poller,
            delay_tracker,
            reader_settings,
            settings_lock,
            active_roi,
            webhook_notifier,
            webhook_delivery_lock,
        ),
    )
    print(f"image-viewer: serving on port {settings.port}; image={settings.image_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()
        if camera_thread is not None:
            camera_thread.join(timeout=2)


def main() -> None:
    serve()


if __name__ == "__main__":
    main()