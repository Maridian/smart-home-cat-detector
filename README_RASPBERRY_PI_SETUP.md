# Raspberry Pi Setup - Simple Guide

Deploy your trained YOLOv8 cat detector to Raspberry Pi 4.

## Prerequisites

- ✅ Trained model: `cat_yolov8n.pt` (from `python main.py train`)
- ✅ Raspberry Pi 4 (4GB or 8GB RAM)
- ✅ Raspberry Pi OS installed
- ✅ SSH enabled on Raspberry Pi
- ✅ Git installed on Raspberry Pi

---

## Quick Setup (4 Steps)

### Step 1: Export Model on your PC

```bash
# Export to ONNX (recommended for Raspberry Pi 4)
python main.py export

# Or choose a specific format:
python src/models/export_model.py --format ncnn    # Fastest (8-12 FPS)
python src/models/export_model.py --format onnx    # Best balance (5-10 FPS)
python src/models/export_model.py --format tflite  # Smallest size (3-7 FPS)
```

This will export your `cat_yolov8n.pt` model.

### Step 2: Clone Repository on Raspberry Pi

```bash
# SSH to Raspberry Pi
ssh pi@raspberrypi.local

# Clone repository
cd ~
git clone https://github.com/YOUR_USERNAME/smart-home-cat-detector.git
cd smart-home-cat-detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-minimal.txt
```

### Step 3: Transfer Exported Model

```bash
# From your PC, transfer only the exported model
# (Replace cat_yolov8n.onnx with your actual exported model)
scp models/trained/cat_yolov8n.onnx pi@raspberrypi.local:~/smart-home-cat-detector/models/trained/
```

> **Note:** All code comes from Git clone, only the trained model needs to be transferred via SCP!

### Step 4: Configure and Run

```bash
# SSH to Raspberry Pi
ssh pi@raspberrypi.local
cd ~/smart-home-cat-detector

# Create/edit .env file
nano .env
```

Add this configuration:

```env
# Model (use your exported model)
MODEL_PATH=models/trained/cat_yolov8n.onnx

# RTSP Camera
RTSP_URL=rtsp://username:password@192.168.1.100:554/stream

# Detection settings
DETECTION_CONFIDENCE=0.30

# Save detections to
IMAGE_SAVE_PATH=/home/pi/cat_detections

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
NOTIFICATION_COOLDOWN=60

# Headless mode (no display needed)
HEADLESS_MODE=1
```

Save with `Ctrl+X`, `Y`, `Enter`.

**Run the detector:**

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

python3 main.py live
```

## That's it! 🎉

Your detector now runs on Raspberry Pi with the same `main.py live` command!

---

---

## Model Format Comparison

| Format | Speed (FPS) | File Size | Compatibility | Recommended |
|--------|-------------|-----------|---------------|-------------|
| **NCNN** | 8-12 | Medium | ARM optimized | ✅ Best for speed |
| **ONNX** | 5-10 | Medium | Universal | ✅ Best balance |
| **TFLite INT8** | 3-7 | Smallest | Mobile/Edge | Good for storage |
| PyTorch (.pt) | 2-5 | Largest | Requires torch | Not recommended |

## Advanced Configuration

### Performance Optimization

```bash
# Export smaller image size for faster inference
python src/models/export_model.py --format ncnn --imgsz 416
# This is ~2x faster but slightly less accurate
```

### Increase GPU Memory (Raspberry Pi)

```bash
sudo raspi-config
# Navigate to: Performance Options → GPU Memory → Set to 128MB
sudo reboot
```

### Increase Swap (for stability)

```bash
sudo nano /etc/dphys-swapfile
# Change: CONF_SWAPSIZE=2048
sudo systemctl restart dphys-swapfile
```

## Update Model on Raspberry Pi

When you retrain your model:

```bash
# On your PC: Export new model
python main.py export

# Transfer to Raspberry Pi (overwrites old model)
scp models/trained/cat_yolov8n.onnx pi@raspberrypi.local:~/smart-home-cat-detector/models/trained/

# Restart detector on Raspberry Pi
ssh pi@raspberrypi.local
sudo systemctl restart cat-detector  # If running as service
# Or just restart: python3 main.py live
```

## Troubleshooting

### Model not found
```bash
# Activate venv first
source .venv/bin/activate

# Check what models you have
ls -la models/trained/

# Export from your PC if needed
python src/models/export_model.py --model runs/detect/train/weights/best.pt --format onnx
```

### Camera connection issues
```bash
# Test RTSP stream
ffplay rtsp://username:password@192.168.1.100:554/stream
```

### Out of memory
```bash
# Increase swap
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=2048
sudo systemctl restart dphys-swapfile
```

## Run as Service (Auto-start on boot)

To run the detector automatically when Raspberry Pi boots:

```bash
# Create service file
sudo nano /etc/systemd/system/cat-detector.service
```

Add this content:

```ini
[Unit]
Description=YOLOv8 Cat Detector Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-home-cat-detector
Environment="HEADLESS_MODE=1"
# Use venv python
ExecStart=/home/pi/smart-home-cat-detector/.venv/bin/python3 /home/pi/smart-home-cat-detector/main.py live
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save with `Ctrl+X`, `Y`, `Enter`.

**Enable and start the service:**

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
sudo systemctl enable cat-detector

# Start service now
sudo systemctl start cat-detector

# Check status
sudo systemctl status cat-detector

# View live logs
sudo journalctl -u cat-detector -f
```

**Manage the service:**

```bash
# Stop detector
sudo systemctl stop cat-detector

# Restart detector
sudo systemctl restart cat-detector

# Disable auto-start
sudo systemctl disable cat-detector

# View all logs
sudo journalctl -u cat-detector --no-pager
```
