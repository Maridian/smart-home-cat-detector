"""Validation script for trained YOLOv8 cat detection model"""
import argparse
from pathlib import Path
from ultralytics import YOLO

from src.utils.config import get_project_root
from src.utils.device import get_device, print_device_info

# Configuration
PROJECT_ROOT = get_project_root()
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "val"
TRAINED_MODELS_DIR = PROJECT_ROOT / "models" / "trained"

DEVICE = get_device()

def main(model_name=None):
    # Determine which model to validate
    if model_name:
        model_path = TRAINED_MODELS_DIR / model_name
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            print(f"\nAvailable models in {TRAINED_MODELS_DIR}:")
            if TRAINED_MODELS_DIR.exists():
                models = sorted(TRAINED_MODELS_DIR.glob("*.pt"))
                for m in models:
                    print(f"  - {m.name}")
            return
    else:
        # Use default latest model
        model_path = TRAINED_MODELS_DIR / "cat_yolov8n.pt"
        print("=== Model Validation ===")
    print(f"Model:   {model_path}")
    print(f"Dataset: {DATA_YAML}")
    print_device_info()
    
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print(f"        Please train the model first: python main.py train")
        return
    
    if not DATA_YAML.exists():
        print(f"[ERROR] Dataset configuration not found: {DATA_YAML}")
        print(f"        Please run auto-labeling first: python main.py label")
        return

            # Load trained model
    print("\nLoading model...")
    model = YOLO(str(model_path))

        # Run validation
    print("Starting validation...\n")
    # Create unique output name based on model version
    model_stem = model_path.stem
    output_name = f"val_{model_stem}"
    
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",           # Evaluates on the 'val' set specified in data.yaml
        imgsz=640,             # Image size used during evaluation
        batch=16,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name=output_name,
        exist_ok=True,
        save_json=True,        # Saves metrics to JSON
        plots=True             # Generates confusion matrix, PR curves, etc.
    )

            # Print summary metrics
    print("\n" + "="*60)
    print("VALIDATION METRICS SUMMARY")
    print("="*60)
    print(f"Model:               {model_path.name}")
    print(f"Precision (P):       {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
    print(f"Recall (R):          {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
    print(f"mAP@0.50:            {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"mAP@0.50-0.95:       {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print("="*60)
    print(f"\nDetailed results saved to:")
    print(f"  {OUTPUT_DIR / output_name}")
    print("\n[✓] Validation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLOv8 cat detection model")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model filename to validate (e.g., cat_yolov8n_20250108_143022.pt). Default: cat_yolov8n.pt (latest)"
    )
    args = parser.parse_args()
    
    try:
        main(model_name=args.model)
    except KeyboardInterrupt:
        print("\n[!] Validation interrupted by user")