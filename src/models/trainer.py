"""Training script for YOLOv8 cat detection model"""
import shutil
import os
from pathlib import Path
from datetime import datetime
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

        # Export best weights with versioning
        best_weights_path = OUTPUT_DIR / "cat_yolov8n" / "weights" / "best.pt"
        trained_models_dir = PROJECT_ROOT / "models" / "trained"
    
        if best_weights_path.exists():
            trained_models_dir.mkdir(parents=True, exist_ok=True)
        
            # Create versioned filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_filename = f"cat_yolov8n_{timestamp}.pt"
            versioned_path = trained_models_dir / versioned_filename
        
            # Save versioned model
            shutil.copy(best_weights_path, versioned_path)
        
            # Update symlink to latest model
            latest_link = trained_models_dir / "cat_yolov8n.pt"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
        
            # Create symlink (cross-platform compatible)
            try:
                os.symlink(versioned_filename, str(latest_link))
            except OSError:
                # Fallback: copy file if symlink not supported (Windows without admin)
                shutil.copy(versioned_path, latest_link)
        
            print("\n" + "="*60)
            print("TRAINING COMPLETE")
            print("="*60)
            print(f"Versioned model saved to:")
            print(f"  {versioned_path}")
            print(f"\nLatest model link:")
            print(f"  {latest_link} -> {versioned_filename}")
            print(f"\nTraining results available at:")
            print(f"  {OUTPUT_DIR / 'cat_yolov8n'}")
            print("="*60)
        else:
            print(f"\n[WARNING] Best weights not found at expected location")
            print(f"          {best_weights_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")