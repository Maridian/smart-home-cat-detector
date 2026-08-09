# Smart Home Cat Detector 🐱

A streamlined Deep Learning system for real-time cat detection using YOLOv8, featuring automated data collection, training, validation, benchmarking, and live detection on RTSP streams or webcams.

## ✨ Features

- **Automated Data Collection**: Capture and save images with cats from RTSP camera streams
- **Auto-Labeling**: Automatically generate YOLO-format labels using pretrained YOLOv8m
- **Custom Training**: Train YOLOv8n model on your collected dataset
- **Model Validation**: Evaluate model performance with detailed metrics
- **Performance Benchmarking**: Measure latency and FPS on CPU and GPU
- **Live Detection**: Real-time cat detection on RTSP streams or webcam
- **Centralized CLI**: Single entry point for all operations

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster training and inference)
- RTSP camera or webcam

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
# RTSP camera URL (or use 0 for webcam)
RTSP_URL=rtsp://username:password@192.168.1.100:554/stream1

# Data collection settings
OUTPUT_DIR=data/raw
SAVE_INTERVAL=0.5
CONFIDENCE_THRESHOLD=0.1

# Detection settings
DETECTION_CONFIDENCE=0.30
```

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
- Shows detection status overlay
- Press 'q' to quit

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
# 1. Collect training images
python main.py collect

# 2. Generate labels automatically
python main.py label

# 3. Train the model
python main.py train

# 4. Validate performance
python main.py validate

# 5. Run live detection
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

## 🏠 Smart Home Integration

Integrate with your smart home system:

- **MQTT**: Publish detection events to a broker
- **REST API**: Wrap detector in FastAPI
- **Home Assistant**: Create automation triggers
- **Node-RED**: Connect to flows

## 🐛 Troubleshooting

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

Adjust in `.env` or code:

```python
CONF_THRESHOLD = 0.25  # Lower = more detections, more false positives
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
