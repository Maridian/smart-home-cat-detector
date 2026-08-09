"""Validation script for trained YOLOv8 cat detection model"""
from pathlib import Path
from ultralytics import YOLO

from src.utils.config import get_project_root
from src.utils.device import get_device, print_device_info

# Configuration
PROJECT_ROOT = get_project_root()
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "cat_yolov8n.pt"
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "val"

DEVICE = get_device()

def main():
    print("=== Model Validation ===")
    print(f"Model:   {MODEL_PATH}")
    print(f"Dataset: {DATA_YAML}")
    print_device_info()
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print(f"        Please train the model first: python main.py train")
        return
    
    if not DATA_YAML.exists():
        print(f"[ERROR] Dataset configuration not found: {DATA_YAML}")
        print(f"        Please run auto-labeling first: python main.py label")
        return

        # Load trained model
    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))

    # Run validation
    print("Starting validation...\n")
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",           # Evaluates on the 'val' set specified in data.yaml
        imgsz=640,             # Image size used during evaluation
        batch=16,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="cat_val_results",
        exist_ok=True,
        save_json=True,        # Saves metrics to JSON
        plots=True             # Generates confusion matrix, PR curves, etc.
    )

        # Print summary metrics
    print("\n" + "="*60)
    print("VALIDATION METRICS SUMMARY")
    print("="*60)
    print(f"Precision (P):       {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
    print(f"Recall (R):          {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
    print(f"mAP@0.50:            {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"mAP@0.50-0.95:       {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print("="*60)
    print(f"\nDetailed results saved to:")
    print(f"  {OUTPUT_DIR / 'cat_val_results'}")
    print("\n[✓] Validation completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Validation interrupted by user")