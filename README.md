# Smart Home Cat Detector 🐱📡

A modular Deep Learning system for real-time cat detection via camera streams (RTSP), featuring automated data collection, training, benchmarking, and integration into smart home environments.

---

## 🚀 Features

* **Real-time Inference:** YOLOv8-based cat detection on GPU, CPU, or edge devices (Raspberry Pi / ONNX).
* **Automated Data Collection:** Frame extraction & motion detection from RTSP camera streams.
* **Pre-Labeling Workflow:** Accelerated annotation process by automatically generating bounding box proposals for Label Studio.
* **Performance Benchmarking:** Integrated latency (ms) and throughput (FPS) analysis across various batch sizes.

---

## 🛠️ Installation & Setup

### 1. Clone the repository & create a virtual environment
```bash
git clone [https://github.com/your-username/smart-home-cat-detector.git](https://github.com/your-username/smart-home-cat-detector.git)
cd smart-home-cat-detector

python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate