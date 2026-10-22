FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system boiler \
    && useradd --system --gid boiler --create-home --home-dir /home/boiler boiler \
    && mkdir -p /data \
    && chown boiler:boiler /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY boiler_reader ./boiler_reader
RUN pip install --no-cache-dir .
# Pre-download EasyOCR models so first container start doesn't fetch them.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

USER boiler
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen('http://127.0.0.1:'+os.getenv('WEB_PORT','8080')+'/', timeout=3)" || exit 1

CMD ["python", "-m", "boiler_reader.image_viewer"]