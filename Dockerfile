# ============================================================
# Smart Home Cat Detector - Live Operation (Raspberry Pi)
#
# Build (directly on the Pi, 64-bit OS = arm64):
#   docker build -t cat-detector:latest .
#
# Run (configuration comes from .env at runtime):
#   docker run --rm --env-file .env \
#       -v /home/admin123/cat_detections:/home/admin123/cat_detections \
#       --network host cat-detector:latest
#   (or: docker compose up -d)
# ============================================================

FROM python:3.10-slim-bullseye

# Only the runtime libraries actually needed by opencv-python-headless.
# FFmpeg/GStreamer codecs are bundled in the headless wheel -> no libgl1,
# no GStreamer, no *-dev packages, no wget required.
#   libglib2.0-0   -> GLib (cv2 core)
#   libgomp1       -> OpenMP (cv2 / ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only the minimal dependencies for live operation (CPU-only, ARM)
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy source code plus trained model into the image.
# (.pt files are excluded via .dockerignore -> only ONNX ends up in the image)
COPY src/ ./src/
COPY main.py .
COPY models/trained/ ./models/trained/

# Storage location for detection images (mounted as volume from the host)
RUN mkdir -p /home/admin123/cat_detections

# Runtime configuration (can be overridden via -e / env_file / compose)
ENV PYTHONUNBUFFERED=1
ENV HEADLESS_MODE=1
ENV DETECTION_CONFIDENCE=0.30

# Run as non-root user (uid 1000 = default user "pi" on the Pi)
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /home/admin123/cat_detections
USER appuser

# Start the live detector (equivalent to: python main.py live)
CMD ["python", "main.py", "live"]
