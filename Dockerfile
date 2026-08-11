FROM python:3.10-slim-bullseye

# System dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY .env* ./

# Create directories for models and USB mount
RUN mkdir -p /app/models/trained /mnt/usb

# Environment variables (can be overridden)
ENV PYTHONUNBUFFERED=1
ENV DETECTION_CONFIDENCE=0.30
ENV RTSP_URL=0

# Run the detector
CMD ["python", "-m", "src.detector.detector"]
