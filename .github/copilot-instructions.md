# Copilot Instructions - Smart Home Cat Detector

## Project Overview
YOLOv8-based real-time cat detection system with RTSP stream support, automated data collection, training, and Telegram notifications. Raspberry Pi-compatible with Docker support.

## Architecture (Token-Optimized)

```
main.py (CLI)
├── src/
│   ├── data_collection/  → collect_data.py (RTSP capture)
│   ├── labeling/         → auto_label.py, custom_dataset.py
│   ├── models/           → trainer.py, detector.py, export_model.py
│   ├── live_monitoring/  → start_monitoring.py (Telegram)
│   ├── benchmark/        → val.py, benchmark.py
│   └── utils/            → config.py, device.py, rtsp_stream.py
├── data/                 → raw/, annotated/{train,val}/, exports/
├── models/               → trained/, exported/
└── runs/                 → detect/, val/
```

**Tech Stack**: PyTorch, YOLOv8 (ultralytics), OpenCV, python-telegram-bot, dotenv

**Deployment**: Docker + Raspberry Pi with USB storage for images

## Core Workflows
1. **collect** → Saves RTSP frames when cat detected (no human)
2. **label** → Auto-labeling with YOLOv8m → 80/20 train/val split
3. **train** → Fine-tune YOLOv8n → `models/trained/cat_yolov8n.pt`
4. **live** → Real-time monitoring + Telegram notification (cooldown-based)

## Development Guidelines

### Code Conventions
- **Comments**: Always English
- **User Output/CLI**: Always German (print statements, error messages)
- **Docstrings**: English, concise
- **Variables**: snake_case, descriptive names
- **Constants**: UPPER_SNAKE_CASE

### Critical Thinking & Communication
- **Question actively**: "Is this approach optimal for embedded devices?"
- **Ask for clarification** on unclear requirements (batch size, paths, config)
- **Consider edge cases**: Raspberry Pi performance, RTSP timeouts, Telegram rate limits
- **Security**: No secrets in code, only .env

### Token Efficiency
- Use abbreviations where appropriate: `cfg` instead of `configuration`
- Avoid repeating explanations of known patterns (YOLO, RTSP)
- In code reviews: Comment only relevant sections
- Leverage context from README/architecture instead of re-explaining

### Technical Preferences
- **Paths**: Always `pathlib.Path` instead of `os.path`
- **Config**: Via `.env` + `load_dotenv()`, never hardcoded
- **Error Handling**: Explicit > implicit (e.g., RTSP timeout handling)
- **Logging**: `print()` for user output (German), consider logging module later
- **Device**: Auto-detect GPU/CPU via `torch.cuda.is_available()`

### Performance Focus
- **YOLOv8n** for Raspi (small, fast)
- **Batch Processing**: Single frames for live detection
- **Image Size**: 640x640 standard, consider 416x416 for Raspi
- **FP16**: For GPU export to enable faster inference

### Testing & Validation
- Before major changes: `python main.py live --debug` for Telegram test
- After training: `python main.py validate` for metrics
- Benchmark on architecture changes: `python main.py benchmark`

### Module-Specific Notes

**src/live_monitoring/start_monitoring.py**:
- Respect cooldown mechanism (NOTIFICATION_COOLDOWN)
- Image save path from .env
- RTSP reconnect logic

**src/models/trainer.py**:
- Epochs, batch_size, imgsz customizable
- Auto-saves best.pt
- Data augmentation via YOLOv8 defaults

**src/utils/rtsp_stream.py**:
- Retry logic for unstable streams
- FPS limiting for CPU efficiency

**Docker**:
- Host network mode for RTSP access
- Volume for `/mnt/usb` (Raspi USB stick)
- .env injected via `environment`

## Common User Requests & Responses

### "Improve model accuracy"
→ Ask: More data? Different augmentation? Increase epochs? Hyperparameter tuning?

### "RTSP not working"
→ Check: URL format, firewall, VLC test, timeout settings in rtsp_stream.py

### "Too many Telegram notifications"
→ Increase NOTIFICATION_COOLDOWN in .env

### "Training too slow"
→ GPU available? CUDA installed? Reduce batch size? Smaller dataset?

### "False positives"
→ Increase DETECTION_CONFIDENCE, collect more negative samples (--force mode)

## Quick Commands (Reference)
```bash
# Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Development cycle
python main.py collect [--force]
python main.py label
python main.py train
python main.py validate [--model filename.pt]
python main.py live [--debug]

# Docker
docker-compose up -d
docker logs -f cat_detector
```

## Anti-Patterns
❌ Hardcoded paths → ✅ pathlib + config.py
❌ Secrets in code → ✅ .env
❌ Blocking I/O without timeout → ✅ RTSP retry logic
❌ German code comments → ✅ English only
❌ Unnecessary dependencies → ✅ Check requirements-minimal.txt

## Optimization Potential (For Discussion)
- [ ] Logging module instead of print()
- [ ] Async RTSP handling (asyncio)
- [ ] Database for detection history (SQLite)
- [ ] Web dashboard (Flask/FastAPI)
- [ ] Multi-camera support
- [ ] TensorRT for Raspi 4

---

**Critical**: When uncertain, ALWAYS ask for clarification instead of making assumptions. User prefers explicit clarification before implicit implementation.
