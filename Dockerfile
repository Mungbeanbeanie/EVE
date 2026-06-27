# EVE — application image.
#
# NOTE: This image is best for TEXT mode and for running the app alongside
# Postgres. VOICE mode needs host microphone/speaker access, which does not
# pass cleanly into a container on macOS — run voice mode on the host instead.

FROM python:3.11-slim

# System deps:
#   ffmpeg        — Whisper uses it to decode/resample audio
#   portaudio19   — headers for building PyAudio
#   gcc/espeak    — build tools + pyttsx3's Linux speech backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        portaudio19-dev \
        gcc \
        espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to text mode (no audio hardware required).
CMD ["python", "main.py", "--mode", "text"]
