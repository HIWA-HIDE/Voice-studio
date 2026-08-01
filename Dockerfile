# ── Zenvyrolabs Voice Studio ──
# Task 1: one image, zero manual setup. Pins Python 3.11 (3.12+ breaks the
# AI libraries per the Engineering Handbook) and bundles FFmpeg so the user
# never has to install Python or FFmpeg themselves.

FROM python:3.11-slim

# FFmpeg is required by pydub for every audio read/export in app.py.
# build-essential + git are needed to build a couple of the ML deps from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached and doesn't get
# re-downloaded (avoiding the "2.4GB download fails mid-transfer" problem)
# every time application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Data that must survive a container restart lives under these paths —
# docker-compose.yml mounts volumes on top of them.
RUN mkdir -p /app/saved_voices /app/rvc_models /app/training_data /app/hf_cache /app/temp

ENV RUNNING_IN_DOCKER=1 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]
