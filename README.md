# Smart Home Cat Detector 🐱

A YOLOv8 cat detector that runs on a Raspberry Pi: it watches an RTSP camera stream, detects cats in real time and sends you a Telegram notification with the annotated image.

## How it works

- Code, Docker setup and the exported model (`models/trained/cat_yolov8n.onnx`) live in this repository — the ONNX model is baked into the Docker image.
- A **self-hosted GitHub Actions runner** on the Raspberry Pi runs the deploy workflow.
- The workflow builds the image directly **on the Pi**, writes the secrets from GitHub into `.env` and starts the container with `docker compose up -d`.
- The container runs with `restart: unless-stopped` → it starts automatically on boot and restarts after crashes.

## Use cases & required steps

Different goals need different steps — here is the overview:

| Use case | Steps | Where |
|---|---|---|
| **Live detection** (model already trained, in the repo) | Raspberry Pi setup → Config → Deploy via GitHub Actions | Tutorial below |
| **Retrain the model** (own data / better accuracy) | `collect` → `label` → `train` → `validate` → `export` → Deploy | PC → Pi |
| **Test / compare performance** (optional) | `train` (or existing model) + `benchmark` + `validate` | PC |
| **Test Telegram** (no camera/stream) | `live --debug` | PC |

All `python main.py <command>` steps run on **your PC** (GPU optional, only speeds up training) — not on the Pi. The Pi only runs the exported ONNX model inside the Docker container.

### Workflow 1 — Live detection (no training, recommended)

The model `models/trained/cat_yolov8n.onnx` is already in the repo. All you need: [Raspberry Pi setup](#raspberry-pi-setup-one-time) → [Configuration](#configuration) → [Deploy via GitHub Actions](#deploy-via-github-actions). No training required.

### Workflow 2: Train a model and roll it out on the Pi

Train your own model on your cat images and then deploy it to the Pi via GitHub Actions:

```bash
# 1. Collect training images (RTSP/webcam; saves only frames with cats)
python main.py collect
#    Option --force: saves ALL frames (for negative samples)

# 2. Auto-label + train/val split (80/20)
python main.py label

# 3. Train YOLOv8n (best weights → models/trained/cat_yolov8n.pt)
python main.py train

# 4. Validate (precision, recall, mAP, confusion matrix, PR curves)
python main.py validate

# 5. Optional: measure inference speed (CPU vs. GPU, various batch sizes)
python main.py benchmark

# 6. Export → overwrites models/trained/cat_yolov8n.onnx
python main.py export
```

After step 6: commit/push the new `cat_yolov8n.onnx` and re-run the **Deploy to Raspberry Pi** workflow (see [Update the model](#update-the-model)).

### Workflow 3 — Benchmark & validation

```bash
# Inference speed (FPS/latency) for various batch sizes, CPU vs. GPU
python main.py benchmark
# → Plot saved under data/exports/benchmark_results.png

# Model quality (precision, recall, mAP, confusion matrix, PR curves)
python main.py validate
# → Results under runs/val/
```

### CLI reference (training PC)

Everything goes through the central entry point `python main.py <command>` — on your PC with `pip install -r requirements.txt` (the Pi only runs the container):

| Command | What it does |
|---|---|
| `collect` | Saves frames from the RTSP/webcam stream when a cat is detected (no humans) → `data/raw/`; `--force` saves all frames (negative samples) |
| `label` | Auto-labels images with a pretrained YOLOv8m, creates an 80/20 split + `data.yaml` → `data/annotated/` |
| `train` | Fine-tunes YOLOv8n on your dataset; best weights → `models/trained/cat_yolov8n.pt` |
| `validate` | Evaluates the model (precision, recall, mAP), generates confusion matrix + PR curves → `runs/val/` |
| `benchmark` | Measures latency/FPS for various batch sizes (CPU vs. GPU) → plot under `data/exports/benchmark_results.png` |
| `export` | Converts `cat_yolov8n.pt` → `models/trained/cat_yolov8n.onnx` (goes into the Docker image) |
| `live` | Live detection (runs automatically on the Pi in the container); `--debug` sends a test notification without a stream |

## Raspberry Pi setup (one-time)

Requirements: Raspberry Pi 4 (4 GB or 8 GB recommended) with **64-bit Raspberry Pi OS**.

### 1. SSH into the Pi

```bash
ssh pi@raspberrypi.local
```

### 2. Update the system

```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Log out and back in (or reboot), then verify:

```bash
docker --version
docker compose version
```

### 4. Install the GitHub Actions runner (Linux, ARM64)

Go to your repository on GitHub: **Settings → Actions → Runners → New self-hosted runner**. Select **Linux / ARM64** and copy the download + configure commands from the dialog:

```bash
mkdir actions-runner && cd actions-runner
# download and extract (command shown in the GitHub dialog)
./config.sh --url https://github.com/YOUR_USERNAME/smart-home-cat-detector --token YOUR_TOKEN
```

Test it once with `./run.sh` (the runner then shows as "idle" in GitHub). For a permanent setup that survives reboots, install it as a service:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

> **Note:** the model `models/trained/cat_yolov8n.onnx` must be in the repository before the image can be built. It is already part of the repo — replace it when you retrain (see [Update the model](#update-the-model)).

## Configuration

The container gets its configuration from two files via `env_file` in `docker-compose.yml` (later entries win on name collisions):

| File | Contents | Examples |
|---|---|---|
| `config.env` (committed) | Non-secret default settings | `DETECTION_CONFIDENCE`, `IMAGE_SAVE_PATH`, `NOTIFICATION_COOLDOWN` |
| `.env` (never committed) | Secrets — created automatically on the Pi by the deploy workflow | `RTSP_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |

> **Important:** detection images are stored at `IMAGE_SAVE_PATH` (default `/home/admin123/cat_detections`). This path must always match the volume mounted in `docker-compose.yml` — otherwise the images are written inside the container and lost when the container is recreated.

### Add the GitHub secrets

In your repository: **Settings → Secrets and variables → Actions → New repository secret**.

| Secret | Description | Example |
|---|---|---|
| `RTSP_URL` | Camera stream URL (`0` for a webcam) | `rtsp://user:pass@192.168.1.100:554/stream` |
| `TELEGRAM_BOT_TOKEN` | From @BotFather | `123456:ABC...` |
| `TELEGRAM_CHAT_ID` | Your chat ID | `123456789` |

**Telegram quick start:** create a bot with **@BotFather** (`/newbot`), send it a message, then read your chat ID from `https://api.telegram.org/bot<TOKEN>/getUpdates` (`"chat":{"id":...}`). The bot must have received at least one message from you.

## Deploy via GitHub Actions

1. Push your code (including the exported model and any `config.env` changes) to GitHub.
2. Open the **Actions** tab → **Deploy to Raspberry Pi** → **Run workflow**.
3. The self-hosted runner on the Pi executes it: checkout → write `.env` from the secrets → `docker compose down` → build image → `docker compose up -d`.
4. The workflow prints the container status and the last 50 log lines — the "Deployment summary" markers show success/failure.

Check on the Pi:

```bash
docker compose ps              # status of the container
docker logs cat_detector --tail 50
docker logs -f cat_detector    # follow live logs
```

## Update the model

1. Export the new model as `models/trained/cat_yolov8n.onnx` (on your PC: `python main.py export`).
2. Commit and push the new file.
3. Re-run the **Deploy to Raspberry Pi** workflow.

> Retraining? Then follow [Workflow 2: Train a model](#workflow-2-train-a-model-and-roll-it-out-on-the-pi): `collect` → `label` → `train` → `validate` → `export` → Deploy. Step 6 (`export`) automatically overwrites the ONNX file from step 1.

## Useful commands

```bash
docker compose ps -a                                      # status
docker logs cat_detector --tail 100                       # recent logs
docker compose down                                       # stop + remove container (image stays)
docker inspect cat_detector --format 'Restarts={{.RestartCount}} ExitCode={{.State.ExitCode}}'   # restart count / exit code
docker exec cat_detector printenv TELEGRAM_CHAT_ID        # what the container actually sees
docker compose run --rm cat-detector python main.py live  # run in foreground to see errors
```

## Troubleshooting

### Container stuck in a restart loop

```bash
docker logs cat_detector --tail 100
docker inspect cat_detector --format 'Restarts={{.RestartCount}} ExitCode={{.State.ExitCode}}'
docker compose run --rm cat-detector python main.py live   # foreground = real error message
```

### Build fails — model not found

The Docker image copies `models/trained/` — if `cat_yolov8n.onnx` is missing or not committed, the build fails. Commit the model file and re-run the workflow.

### Telegram: "chat not found" (HTTP 400)

Detection works but no notification arrives:

1. Press **Start** in your Telegram chat with the bot — bots cannot message users who never started them.
2. Verify the chat ID via `https://api.telegram.org/bot<TOKEN>/getUpdates` (group IDs are negative, supergroups start with `-100`).
3. Check what the container really has: `docker exec cat_detector printenv TELEGRAM_CHAT_ID` (no quotes or whitespace).

### Camera/RTSP not reachable

The container uses `network_mode: host`, so it can reach local cameras directly. Verify the RTSP URL (`rtsp://user:pass@ip:554/stream`) with VLC or `ffprobe` on the Pi.

---

**Happy Cat Detecting! 🐱**