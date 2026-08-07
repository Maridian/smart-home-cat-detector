import sys
import torch
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ultralytics import YOLO

MODEL_PATH = PROJECT_ROOT / "cat_yolov8n.pt"
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "val"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

def main():
    if not MODEL_PATH.exists():
        print(f"[Error] Model file not found at: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading trained model from: {MODEL_PATH}")
    print(f"Evaluating on dataset defined in: {DATA_YAML}")

    # Load your custom trained model
    model = YOLO(str(MODEL_PATH))

    # Run validation
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

    # Print summary metrics to console
    print("\n" + "=" * 45)
    print("        EVALUATION METRICS SUMMARY        ")
    print("=" * 45)
    print(f" Precision    (P):    {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
    print(f" Recall       (R):    {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
    print(f" mAP @ 0.50:          {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f" mAP @ 0.50-0.95:     {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print("=" * 45)
    print(f"\n[✓] Detailed plots & results saved to: {OUTPUT_DIR / 'cat_val_results'}")

if __name__ == "__main__":
    main()