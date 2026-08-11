# Smart Home Cat Detector 🐱

A streamlined Deep Learning system for real-time cat detection using YOLOv8, featuring automated data collection, training, validation, benchmarking, and live detection with Telegram notifications.

## ✨ Features

- **Automated Data Collection**: Capture and save images with cats from RTSP camera streams
- **Auto-Labeling**: Automatically generate YOLO-format labels using pretrained YOLOv8m
- **Custom Training**: Train YOLOv8n model on your collected dataset
- **Model Validation**: Evaluate model performance with detailed metrics
- **Performance Benchmarking**: Measure latency and FPS on CPU and GPU
- **Live Detection**: Real-time cat detection on RTSP streams or webcam
- **Telegram Notifications**: Get instant notifications with images when cats are detected
- **Centralized CLI**: Single entry point for all operations

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster training and inference)
- RTSP camera or webcam
- Telegram account (for notifications)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/smart-home-cat-detector.git
cd smart-home-cat-detector

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
# Camera source (0 for webcam, or RTSP URL)
RTSP_URL=0

# Detection confidence threshold (0.0 - 1.0)
DETECTION_CONFIDENCE=0.60

# Telegram Bot Token (get from @BotFather)
TELEGRAM_BOT_TOKEN=your_token_here

# Telegram Chat ID (get from bot API)
TELEGRAM_CHAT_ID=your_chat_id_here

# Path where detection images are saved
IMAGE_SAVE_PATH=./detections

# Cooldown between notifications (seconds)
NOTIFICATION_COOLDOWN=60
```

### 3. Setup Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow instructions
3. Copy the **Bot Token** to your `.env`
4. Send a message to your bot
5. Get your **Chat ID** from: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
6. Add the Chat ID to your `.env`

## 📖 Usage

All functionality is accessible through the central `main.py` script:

```bash
python main.py <command>
```

### Available Commands

#### 1. Collect Data
Collect training images from your camera:

```bash
python main.py collect
```

- Captures frames when a cat is detected (no humans present)
- Saves images to `data/raw/`
- Press 'q' to stop collection

#### 2. Auto-Label Data
Automatically generate YOLO labels for collected images:

```bash
python main.py label
```

- Uses pretrained YOLOv8m to detect cats
- Creates train/val split (80/20)
- Generates `data.yaml` configuration
- Outputs to `data/annotated/`

#### 3. Train Model
Train a custom YOLOv8n model:

```bash
python main.py train
```

- Fine-tunes YOLOv8n on your dataset
- Saves best weights to `models/trained/cat_yolov8n.pt`
- Training logs in `runs/detect/cat_yolov8n/`

#### 4. Validate Model
Evaluate model performance:

```bash
python main.py validate
```

- Computes precision, recall, mAP metrics
- Generates confusion matrix and PR curves
- Results saved to `runs/val/cat_val_results/`

#### 5. Benchmark Performance
Measure inference speed:

```bash
python main.py benchmark
```

- Tests different batch sizes
- Compares CPU vs GPU performance
- Saves plots to `data/exports/benchmark_results.png`

#### 6. Live Detection
Run real-time detection:

```bash
python main.py live
```

- Displays live video with bounding boxes
- Sends Telegram notification with image when cat is detected
- Respects cooldown period to prevent spam
- Press 'q' to quit

#### 7. Test Notifications
Test Telegram without live stream:

```bash
python main.py live --debug
```

- Sends a test notification to verify setup
- No camera/stream required
- Perfect for testing your configuration

## 📁 Project Structure

```
smart-home-cat-detector/
├── main.py                     # Central CLI entry point
├── requirements.txt            # Python dependencies
├── .env                        # Configuration (not in git)
├── .gitignore
├── README.md
│
├── data/
│   ├── raw/                    # Collected images
│   ├── annotated/              # Labeled dataset
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── data.yaml
│   └── exports/                # Benchmark plots, previews
│
├── models/
│   ├── trained/                # Your trained models
│   └── exported/               # ONNX/TensorRT exports
│
├── runs/
│   ├── detect/                 # Training logs
│   └── val/                    # Validation results
│
└── src/
    ├── benchmark/
    │   ├── benchmark.py        # Performance testing
    │   └── val.py              # Model validation
    ├── capture/
    │   └── rtsp_stream.py      # RTSP stream handler
    ├── data_collection/
    │   └── collect_data.py     # Data collection from RTSP
    ├── dataset/
    │   ├── auto_label.py       # Auto-labeling
    │   └── custom_dataset.py   # Dataset utilities
    ├── live_detector/
    │   └── live_detector.py    # Live detection
    ├── models/
    │   ├── detector.py         # Detection logic
    │   └── trainer.py          # Training logic
    └── utils/
        ├── config.py           # Configuration helpers
        └── device.py           # Device selection
```

## 🔄 Complete Workflow

```bash
# 1. Setup Telegram bot (see Configuration section)

# 2. Test notifications
python main.py live --debug

# 3. Collect training images (optional - if using pretrained model, skip to step 6)
python main.py collect

# 4. Generate labels automatically
python main.py label

# 5. Train the model
python main.py train

# 6. Validate performance (optional)
python main.py validate

# 7. Run live detection
python main.py live

# Optional: Benchmark performance
python main.py benchmark
```

## 🎯 Model Export

### Export to ONNX (for edge devices)

```python
from ultralytics import YOLO

model = YOLO('models/trained/cat_yolov8n.pt')
model.export(format='onnx', imgsz=640)
```

### Export to TensorRT (for NVIDIA devices)

```python
model.export(format='engine', device=0, half=True)
```

## 📱 Telegram Notifications

When a cat is detected, you receive:

- 🐱 **Instant notification** on your phone
- 📸 **Image with bounding box** showing the detection
- 📊 **Confidence score** (e.g., 0.85)
- ⏰ **Timestamp** of detection

### Notification Cooldown

To prevent spam when a cat stays in view, notifications respect a cooldown period (default: 60 seconds). Adjust in `.env`:

```bash
NOTIFICATION_COOLDOWN=300  # 5 minutes
```

## 🐛 Troubleshooting

### Telegram Not Working

```bash
# Test your bot token
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe

# Verify Chat ID
curl https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates

# Test notification
python main.py live --debug
```

### RTSP Connection Issues

```bash
# Test your RTSP stream with VLC
vlc rtsp://username:password@camera_ip:554/stream1

# Try webcam instead
# Set in .env: RTSP_URL=0
```

### Low Detection Accuracy

- Collect more diverse training data (different angles, lighting)
- Increase training epochs in `src/models/trainer.py`
- Adjust confidence threshold in `.env`

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 Advanced Configuration

### Training Hyperparameters

Edit `src/models/trainer.py`:

```python
EPOCHS = 50          # Increase for better accuracy
BATCH_SIZE = 16      # Increase if you have more VRAM
LEARNING_RATE = 0.01 # Adjust learning rate
IMG_SIZE = 640       # Image size for training
```

### Detection Confidence

Adjust in `.env`:

```bash
DETECTION_CONFIDENCE=0.25  # Lower = more detections, more false positives
```

### Image Storage Path

Configure where detection images are saved:

```bash
# Raspberry Pi with USB drive
IMAGE_SAVE_PATH=/mnt/usb/cat_detections

# Windows
IMAGE_SAVE_PATH=C:/Users/YourName/cat_detections

# Relative path (in project folder)
IMAGE_SAVE_PATH=./detections
```

## 📊 Performance Metrics

Typical performance on YOLOv8n:

- **CPU (Intel i7)**: ~30-50 FPS
- **GPU (RTX 3060)**: ~200-300 FPS
- **Raspberry Pi 4**: ~5-10 FPS (with optimization)

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Cat Detecting! 🐱**
