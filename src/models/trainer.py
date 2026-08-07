import sys
import torch
import shutil
from pathlib import Path

# Ensure the project root is in sys.path when executed externally
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ultralytics import YOLO

# Define paths relative to project root
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect"

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def main():
    print(f"Target device: {DEVICE}")
    print(f"Loading dataset configuration from: {DATA_YAML}")

    # 1. Initialize pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # 2. Execute training pipeline
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        imgsz=640,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="cat_yolov8n",
        exist_ok=True,
        workers=4,
    )

    # 3. Export best weights to project root directory
    best_weights_path = OUTPUT_DIR / "cat_yolov8n" / "weights" / "best.pt"
    export_target_path = PROJECT_ROOT / "models" / "trained" / "cat_yolov8n.pt"
    
    if best_weights_path.exists():
        shutil.copy(best_weights_path, export_target_path)
        print(f"\n[✓] Model successfully exported to: {export_target_path}")

if __name__ == "__main__":
    main()