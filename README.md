# Boiler Reader

Camera-based digital boiler pressure gauge reader for Reolink RTSP cameras. The display is located automatically, read by a vision LLM running on a local Ollama server, and reported through Home Assistant webhook notifications and a web UI.

## Reading Pipeline

1. **Grab:** the server pulls a frame from `RTSP_URL` every `POLL_SECONDS`.
2. **Locate:** `locator.py` finds the self-illuminated digit row by brightness thresholding (`BRIGHTNESS_THRESHOLD`, default 30), excludes the OSD timestamp strip and the unit indicator lights, and returns the display ROI. No manual ROI or digit-box configuration is needed.
3. **Read (LLM):** `ocr.py` crops the display, converts it to grayscale, upscales it 4x, and sends it to Ollama's `/api/generate` endpoint with a prompt asking for only the digits. The model's reply is reduced to the first run of 2–3 digits and interpreted by digit count:

   | Reply | Meaning | Example |
   | --- | --- | --- |
   | 2 digits | Pressure, `n.n` bar | `12` → 1.2 bar |
   | 3 digits | Temperature, `nn.n` °C | `321` → 32.1 °C |
   | 3 characters matching a manual error code | Error | `ECS`; `FIL` is reported as `FILL` |
   | anything else | Invalid reading | |

4. **Alert:** low pressure or an invalid/error display that persists past the error delay triggers the Home Assistant webhook (see below).

### LLM requirements

The container does not bundle any OCR or ML libraries (EasyOCR and PyTorch are no longer used). It needs network access to an Ollama server with a vision-capable model pulled, for example:

```sh
ollama pull lfm3-vl:latest
```

| Setting | Env var (seed) | Default |
| --- | --- | --- |
| Ollama URL, e.g. `http://192.168.0.26:11434` | `OLLAMA_URL` | none (required) |
| Model | `OLLAMA_MODEL` | `lfm3-vl:latest` |

Both can also be set in the UI's Admin panel; saved values in `reader-settings.json` take priority over the environment. If `OLLAMA_URL` is not set, readings fail with a configuration error. If the Ollama request fails or times out (60 seconds), the reading is treated as invalid rather than reusing an old value.

Implementation notes: requests use `keep_alive: -1` so the model stays loaded, and a random seed plus a single random pixel in each image prevent Ollama from replaying a cached response for a near-identical frame. Each reading logs the raw LLM reply and inference time to stdout.

### Capturing misreads

Use **Capture Misread** in the UI, entering the value shown on the real display, to save the full frame to `screenshots/` beside `LAST_IMAGE_PATH`. The filename records the LLM's reading and the manual value (`<timestamp>_LLM_<reading>_MANUAL_<value>_<unit>.jpg`), and a matching `_ocr.jpg` file holds the exact crop the LLM was shown. Use these to diagnose misreads and to tune the model, prompt, or brightness threshold.

## Segment Decoder (manual ROI mode)

The original seven-segment decoder is still used when a manual ROI and digit boxes are configured, and by the `decode_image` CLI described below. It crops the configured ROI, applies grayscale CLAHE, Otsu thresholding (using a fallback for uniform images), and light morphological cleanup. Digit boxes are coordinates relative to the cropped ROI. Within each box, normalized sample zones read segments in `a` through `g` order. With `expected_format: auto`, a lit decimal point and two populated numeric positions identify a pressure reading (`n.n`); three identify a temperature (`nn.n`). A blank leading slot is allowed on a three-position display. Three populated positions without a lit decimal point are not accepted as numeric readings. Common seven-segment letters and dashes are recognized; unsupported patterns remain undecoded.

Recognized manual error codes include A10, A11, A20, A21, A30, A40, A41, A42, A58, A59, A60, A70, A71, A80, A81, A90, A91, ECS, SES, and FIL. The three-character display `FIL` is reported as the manual code `FILL`; the raw display text remains available separately.

Install the project and test dependencies with Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest
```

The tests render digits and invalid patterns in memory, exercise a synthetic JPEG, and decode the supplied `tests/boiler.PNG` frame plus raw pressure and temperature camera frames in `tests/fixtures/`. To test another camera image, use the decoder CLI and supply the full-image ROI plus one box per character (coordinates inside boxes are relative to the ROI):

```powershell
py -3.12 -m boiler_reader.decode_image .\tests\fixtures\gauge.jpg `
    --roi 100,80,360,160 `
    --digit-box 10,10,100,140 `
    --digit-box 130,10,100,140 `
    --digit-box 250,10,100,140 `
    --decimal-box 108,136,14,12
```

The command prints JSON containing `value`, `valid`, `unit`, `raw_pattern`, `confidence`, and `display_text`. Use `--format dd.d` to require a temperature reading. Adjust the ROI, character boxes, and decimal-point box for your frame; `nn.n` requires three `--digit-box` arguments. Add real camera images to `tests/fixtures/` for repeatable local calibration and regression checks; no real camera image is included yet.

## Latest Frame Viewer

On Windows, install the project once and double-click `start-server.bat` to launch the viewer. It uses `tests/boiler.PNG` if `LAST_IMAGE_PATH` is not set in the process or `.env`.

Run the latest-frame viewer with:

```powershell
$env:LAST_IMAGE_PATH = "C:\path\to\latest.jpg"
$env:WEB_PORT = "8080"
$env:ROI_X = "255"
$env:ROI_Y = "230"
$env:ROI_W = "170"
$env:ROI_H = "110"
$env:DIGIT_BOXES = "15,10,34,69;53,10,49,76;108,15,45,76"
$env:DECIMAL_BOX = "88,76,12,10"
py -3.12 -m boiler_reader.image_viewer
```

Open `http://localhost:8080/`. The page refreshes every two seconds and shows the latest image with the ROI outlined on the left and the decoded pressure or temperature, display text, segment pattern, and confidence on the right. Configure three `DIGIT_BOXES` entries to read a `nn.n` temperature display. Drag the ROI border to move it; drag the lower-right handle to resize it. Releasing the pointer rescans the current image immediately, updates the sidebar, and saves the ROI. Use **Grab new frame** to request an immediate camera capture; **Last frame** shows the timestamp of the latest grab. **Send test HA message** asks for confirmation, then sends a simulated `ECS` error through the configured webhook and may trigger the HA mobile notification automation. The saved ROI is loaded on restart from `ROI_STATE_PATH`, defaulting to a `roi.json` sidecar beside `LAST_IMAGE_PATH` (for example, `/data/roi.json`).

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

The repository includes `Dockerfile` and `docker-compose.yml` for Portainer. A GitHub Actions workflow publishes multi-architecture images (`linux/amd64` and `linux/arm64`) to `ghcr.io/tc-fpv/boilerwatch` on each push to `master`, tagged `latest` and with the commit SHA. It uses the workflow's scoped `GITHUB_TOKEN`; no registry PAT is stored in the repository. The image uses Python 3.12-slim, runs as a non-root user, and stores the latest frame, camera status, ROI, saved settings, and misread screenshots in the persistent `boilerwatch_data` volume mounted at `/data`. Because OCR is delegated to Ollama, the image contains only OpenCV, NumPy, and python-dotenv, so it is small; the Ollama server must be reachable from the container.

The compose file also runs a small `image-pruner` service that removes dangling images hourly (`docker image prune -f`), so superseded `latest` images do not accumulate on the Portainer host. It mounts the Docker socket, which gives that container root-equivalent access to the host; remove the service if you prefer to prune manually. The `boilerwatch` service's logs are capped at three 10 MB files.

1. Push to `master` and wait for **Actions** > **Publish container** to complete.
2. In GitHub, open the published `boiler-reader` package under the account's Packages. Packages are private by default unless visibility is changed. Make it public for anonymous Portainer pulls, or configure Portainer with a registry credential that has `read:packages` access.
3. In Portainer, choose **Stacks** > **Add stack** > **Git Repository**.
4. Enter the repository URL (`https://github.com/tc-fpv/boiler-reader`), select branch `master`, and set the Compose path to `docker-compose.yml`.
5. Add the variables from `.env.example` in Portainer's stack environment section. Set `RTSP_URL` and `OLLAMA_URL` (and optionally `OLLAMA_MODEL`), and keep `RTSP_USER`, `RTSP_PASS`, and `HA_WEBHOOK_URL` private there. The geometry variables (`ROI_X`, `ROI_Y` etc.) are no longer needed — the display is located automatically. Do not commit `.env` or the webhook ID.
6. Deploy the stack. Portainer pulls `ghcr.io/tc-fpv/boilerwatch:latest`; the viewer is exposed on `WEB_PORT` (default `8080`). Set `IMAGE_TAG` to a `sha-...` tag to pin a specific build.

For local Docker Compose, copy `.env.example` to `.env`, fill in the private values, then run `docker compose up -d`. The app accepts `LOWER_LIMIT` as a compatibility alias, but `LOW_THRESHOLD` is the canonical setting name. The GHCR image is available after the GitHub Actions publish workflow completes.