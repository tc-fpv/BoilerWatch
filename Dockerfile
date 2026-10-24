FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_PORT=8080 \
    EASYOCR_MODULE_PATH=/data/.EasyOCR

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system boilerwatch \
    && useradd --system --gid boilerwatch --create-home --home-dir /home/boilerwatch boilerwatch \
    && mkdir -p /data \
    && chown boilerwatch:boilerwatch /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY boiler_reader ./boiler_reader
# Install CPU-only PyTorch and torchvision together so versions are compatible.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

USER boilerwatch
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen('http://127.0.0.1:'+os.getenv('WEB_PORT','8080')+'/', timeout=3)" || exit 1

CMD ["python", "-m", "boiler_reader.image_viewer"]