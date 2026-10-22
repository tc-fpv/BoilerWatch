# Boiler Reader

Camera-based digital boiler pressure gauge reader for Reolink RTSP cameras, with seven-segment decoding, Home Assistant webhook notifications, and an interactive calibration viewer.

## Decoder Stage

The decoder crops the configured ROI, applies grayscale CLAHE, Otsu thresholding (using a fallback for uniform images), and light morphological cleanup. Digit boxes are coordinates relative to the cropped ROI. Within each box, normalized sample zones read segments in `a` through `g` order. With `expected_format: auto`, a lit decimal point and exactly two populated numeric positions identify a pressure reading (`n.n`); a blank leading slot is allowed on a three-position display. Three populated positions are treated as an error code, even if all three glyphs look numeric. Common seven-segment letters and dashes are recognized; unsupported patterns remain undecoded.

Recognized manual error codes include A10, A11, A20, A21, A30, A40, A41, A42, A58, A59, A60, A70, A71, A80, A81, A90, A91, ECS, SES, and FIL. The three-character display `FIL` is reported as the manual code `FILL`; the raw display text remains available separately.

Install the project and test dependencies with Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest
```

The tests render digits and invalid patterns in memory, exercise a synthetic JPEG, and decode the supplied `tests/boiler.PNG` frame. To test another camera image, use the decoder CLI and supply the full-image ROI plus one box per character (coordinates inside boxes are relative to the ROI):

```powershell
py -3.12 -m boiler_reader.decode_image .\tests\fixtures\gauge.jpg `
    --roi 100,80,360,160 `
    --digit-box 10,10,100,140 `
    --digit-box 130,10,100,140 `
    --digit-box 250,10,100,140 `
    --decimal-box 108,136,14,12
```

The command prints JSON containing `value`, `valid`, `raw_pattern`, `confidence`, and `display_text`. Adjust the ROI, character boxes, and decimal-point box for your frame. Add real camera images to `tests/fixtures/` for repeatable local calibration and regression checks; no real camera image is included yet.

## Latest Frame Viewer

On Windows, install the project once and double-click `start-server.bat` to launch the viewer. It uses `tests/boiler.PNG` if `LAST_IMAGE_PATH` is not set in the process or `.env`.

Run the latest-frame viewer with:

```powershell
$env:LAST_IMAGE_PATH = "C:\path\to\latest.jpg"
$env:WEB_PORT = "8080"
$env:ROI_X = "330"
$env:ROI_Y = "238"
$env:ROI_W = "110"
$env:ROI_H = "94"
$env:DIGIT_BOXES = "0,0,22,85;30,3,48,76"
$env:DECIMAL_BOX = "13,68,9,8"
py -3.12 -m boiler_reader.image_viewer
```

Open `http://localhost:8080/`. The page refreshes every two seconds and shows the latest image with the ROI outlined on the left and the decoder's pressure, display text, segment pattern, and confidence on the right. Drag the ROI border to move it; drag the lower-right handle to resize it. Releasing the pointer rescans the current image immediately, updates the sidebar, and saves the ROI. Use **Grab new frame** to request an immediate camera capture; **Last frame** shows the timestamp of the latest grab. **Send test HA message** asks for confirmation, then sends a simulated `ECS` error through the configured webhook and may trigger the HA mobile notification automation. The saved ROI is loaded on restart from `ROI_STATE_PATH`, defaulting to a `roi.json` sidecar beside `LAST_IMAGE_PATH` (for example, `/data/roi.json`).

When `RTSP_URL` is configured, the server grabs a frame before it starts serving the UI, then grabs another every `POLL_SECONDS`. A failed grab removes the previous frame, writes `camera-status.json`, and is shown as a camera error; old frames are not presented as current. Saving Poll seconds in the sidebar updates the running poller and persists across restarts.

The sidebar's Error delay setting (`ALERT_DELAY_MINUTES`, default 5) tracks how long low pressure or an invalid/error display has continued. Switching between those issue types keeps the timer running; a valid reading at or above the low limit resets it. Set `HA_WEBHOOK_URL` to `http://HOME_ASSISTANT_HOST:8123/api/webhook/WEBHOOK_ID` to POST one event after the delay expires. `HA_WEBHOOK_TIMEOUT` defaults to 5 seconds. A failed POST is retried at the next poll; a successful POST is not repeated until the issue clears and occurs again. The container does not call a Home Assistant notify service.

Create a Home Assistant automation using the same webhook ID as `HA_WEBHOOK_URL`, then replace the mobile notify service with your app's service name:

```yaml
alias: Boiler reader alert
trigger:
    - platform: webhook
        webhook_id: !secret boiler_reader_webhook_id
        allowed_methods:
            - POST
        local_only: true
condition:
    - condition: template
        value_template: "{{ trigger.json.status != 'ok' }}"
action:
    - service: notify.mobile_app_your_device
        data:
            title: Boiler pressure alert
            message: >-
                {{ trigger.json.status | upper }}
                {% if trigger.json.pressure is number %}
                - {{ trigger.json.pressure }} bar
                {% endif %}
                {% if trigger.json.error_code %}
                - code {{ trigger.json.error_code }}
                {% endif %}
                {% if trigger.json.error %}
                - {{ trigger.json.error }}
                {% endif %}
mode: queued
```

Store the webhook ID as `boiler_reader_webhook_id` in Home Assistant `secrets.yaml`; use that same ID in the local `.env` URL. Keep the URL private because anyone with it can trigger the automation.

To test the automation manually, run this standalone command (it is not part of the normal test suite). The default is a dry run; `--send` posts a simulated `ECS` error and can trigger a real mobile notification:

```powershell
py -3.12 -m boiler_reader.send_test_webhook
py -3.12 -m boiler_reader.send_test_webhook --send
```

The sidebar also shows editable poll seconds and low-pressure limit fields. These initialize from `POLL_SECONDS` and `LOW_THRESHOLD` in `.env` (defaults: 30 seconds and 1.0 bar). Select **Save settings** to write them to `READER_SETTINGS_PATH`, defaulting to a `reader-settings.json` sidecar beside `LAST_IMAGE_PATH`; saved JSON values take priority over `.env` on restart. This lets you remove the two values from `.env` after saving. Both sidecar files are ignored by Git. `WEB_PORT` is configurable. ROI defaults match the captured 640×480 Reolink frame and can be overridden with `ROI_X`, `ROI_Y`, `ROI_W`, and `ROI_H`. Character boxes use `DIGIT_BOXES` as semicolon-separated `x,y,width,height` values relative to the ROI; they remain fixed within the resized crop. `DECIMAL_BOX` uses the same coordinate format. Defaults match the captured Reolink frame. The default image path is `/data/last.jpg` for container deployment. Ensure that path and its parent directory are writable by the viewer.

## Portainer Deployment

The repository root contains `Dockerfile` and `docker-compose.yml` for a Git-based Portainer stack. The image uses Python 3.12-slim, runs as a non-root user, and stores the latest frame, camera status, ROI, and saved settings in the persistent `boiler_reader_data` volume mounted at `/data`.

1. In Portainer, choose **Stacks** > **Add stack** > **Git Repository**.
2. Enter `https://github.com/tc-fpv/boiler-reader`, select branch `master`, and set the Compose path to `docker-compose.yml`.
3. Add the variables from `.env.example` in Portainer's stack environment section. Set `RTSP_URL`, and keep `RTSP_USER`, `RTSP_PASS`, and `HA_WEBHOOK_URL` private there. Do not commit `.env` or the webhook ID.
4. Deploy the stack. Portainer builds the image from the repository's Dockerfile; the viewer is exposed on `WEB_PORT` (default `8080`).

For local Docker Compose, copy `.env.example` to `.env`, fill in the private values, then run `docker compose up -d --build`. The app accepts `LOWER_LIMIT` as a compatibility alias, but `LOW_THRESHOLD` is the canonical setting name.