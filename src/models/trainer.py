"""Training script for YOLOv8 cat detection model"""
import shutil
from pathlib import Path
from ultralytics import YOLO

from src.utils.config import get_project_root
from src.utils.device import get_device, print_device_info

# Configuration
PROJECT_ROOT = get_project_root()
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect"

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
IMG_SIZE = 640
MODEL_NAME = "yolov8n.pt"
DEVICE = get_device()

def main():
    print("=== YOLOv8 Training Configuration ===")
    print(f"Model:          {MODEL_NAME}")
    print(f"Dataset:        {DATA_YAML}")
    print(f"Epochs:         {EPOCHS}")
    print(f"Batch Size:     {BATCH_SIZE}")
    print(f"Learning Rate:  {LEARNING_RATE}")
    print(f"Image Size:     {IMG_SIZE}")
    print_device_info()
    
    if not DATA_YAML.exists():
        print(f"[ERROR] Dataset configuration not found: {DATA_YAML}")
        print("        Please run auto-labeling first: python main.py label")
        return

        # Initialize pretrained model
    print(f"\nLoading pretrained model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

        # Start training
    print("\nStarting training...\n")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="cat_yolov8n",
        exist_ok=True,
        workers=4,
        patience=10,  # Early stopping patience
        save=True,
        plots=True
    )

        # Export best weights
    best_weights_path = OUTPUT_DIR / "cat_yolov8n" / "weights" / "best.pt"
    export_target_path = PROJECT_ROOT / "models" / "trained" / "cat_yolov8n.pt"
    
    if best_weights_path.exists():
        export_target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights_path, export_target_path)
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Best weights saved to:")
        print(f"  {export_target_path}")
        print(f"\nTraining results available at:")
        print(f"  {OUTPUT_DIR / 'cat_yolov8n'}")
        print("="*50)
    else:
        print(f"\n[WARNING] Best weights not found at expected location")
        print(f"          {best_weights_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")