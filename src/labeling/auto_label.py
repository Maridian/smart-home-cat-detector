"""Automatic labeling of cat images using YOLOv8m pretrained model"""
import shutil
import random
from pathlib import Path
import cv2
from ultralytics import YOLO

from src.utils.config import get_project_root

# Configuration
PROJECT_ROOT = get_project_root()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"
MANUAL_REVIEW_DIR = ANNOTATED_DIR / "manual_review"

TRAIN_RATIO = 0.8  # 80% train, 20% validation
COCO_CAT_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.10

def setup_directories():
    """Creates the required folder structure."""
    for split in ["train", "val"]:
        (ANNOTATED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (ANNOTATED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

def create_data_yaml():
    """Generates the data.yaml file required for YOLO training."""
    yaml_content = f"""# Dataset configuration for YOLO training
path: {ANNOTATED_DIR.resolve()}
train: train/images
val: val/images

names:
  0: cat

nc: 1  # number of classes
"""
    yaml_path = ANNOTATED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[✓] Created data.yaml at {yaml_path}")

def main():
    print("=== Automatic Labeling Process ===")
    print(f"Input directory:  {RAW_DIR}")
    print(f"Output directory: {ANNOTATED_DIR}")
    print(f"Train/Val split:  {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}\n")
    
    setup_directories()
    create_data_yaml()

    print("\nLoading YOLOv8m model for auto-labeling...")
    model = YOLO("yolov8m.pt")

    # Collect all image files
    image_paths = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.jpeg")) + list(RAW_DIR.glob("*.png"))
    
    if not image_paths:
        print(f"[!] No images found in {RAW_DIR}")
        print("    Please run data collection first: python main.py collect")
        return

    print(f"Found {len(image_paths)} images. Starting auto-labeling...\n")

    # Randomize order and split into train / val
    random.shuffle(image_paths)
    split_index = int(len(image_paths) * TRAIN_RATIO)
    
    datasets = {
        "train": image_paths[:split_index],
        "val": image_paths[split_index:]
    }

    processed_count = 0
    cat_detected_total = 0
    images_with_cats = 0
    preview_saved = False

    for split, paths in datasets.items():
        print(f"\nProcessing {split} set ({len(paths)} images)...")
        for img_path in paths:
            # Run inference
            results = model(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            
            label_lines = []
            cats_in_image = 0
            
            # Filter bounding boxes and convert to YOLO format
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id == COCO_CAT_CLASS_ID:
                    x, y, w, h = box.xywhn[0].tolist()
                    # Class 0 represents 'cat' in the target dataset
                    label_lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    cats_in_image += 1

            # Track statistics
            if cats_in_image > 0:
                images_with_cats += 1
                cat_detected_total += cats_in_image

            # Save preview of first detection
            if cats_in_image > 0 and not preview_saved:
                preview_path = EXPORTS_DIR / "preview_detection.jpg"
                results.save(filename=str(preview_path))
                print(f"  [✓] Saved preview image to: {preview_path}")
                preview_saved = True

            # Save annotated image to manual review folder (only cats)
            review_img_path = MANUAL_REVIEW_DIR / img_path.name
            img = cv2.imread(str(img_path))
            
            # Draw only cat bounding boxes
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id == COCO_CAT_CLASS_ID:
                    # Get box coordinates in pixel values
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"cat {confidence:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.imwrite(str(review_img_path), img)

            # Copy image to target path
            target_img_path = ANNOTATED_DIR / split / "images" / img_path.name
            shutil.copy(img_path, target_img_path)

            # Write label text file
            target_label_path = ANNOTATED_DIR / split / "labels" / f"{img_path.stem}.txt"
            with open(target_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            processed_count += 1

    print("\n" + "="*50)
    print("AUTO-LABELING SUMMARY")
    print("="*50)
    print(f"Total images processed:  {processed_count}")
    print(f"Images with cats:        {images_with_cats} ({images_with_cats/processed_count*100:.1f}%)")
    print(f"Total cats detected:     {cat_detected_total}")
    print(f"Average cats per image:  {cat_detected_total/images_with_cats if images_with_cats > 0 else 0:.2f}")
    print(f"\nDataset location: {ANNOTATED_DIR}")
    print(f"Config file:      {ANNOTATED_DIR / 'data.yaml'}")
    print(f"Manual review:    {MANUAL_REVIEW_DIR}")
    print("\n[i] Check the manual_review folder to verify labels visually.")
    print("="*50)

if __name__ == "__main__":
    main()
