import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# Define paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"

# Training/Validation split ratio (80% train, 20% val)
TRAIN_RATIO = 0.8
COCO_CAT_CLASS_ID = 15  # COCO Class-ID for 'cat'

def setup_directories():
    """Creates the required folder structure."""
    for split in ["train", "val"]:
        (ANNOTATED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (ANNOTATED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

def create_data_yaml():
    """Generates the data.yaml file required for YOLO training."""
    yaml_content = f"""path: {ANNOTATED_DIR.resolve()}
train: train/images
val: val/images

names:
  0: cat
"""
    yaml_path = ANNOTATED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

def main():
    setup_directories()
    create_data_yaml()

    print("Loading YOLOv8m model...")
    model = YOLO("yolov8m.pt")

    # Collect all image files from data/raw
    image_paths = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.jpeg"))
    
    if not image_paths:
        print(f"No images found in {RAW_DIR}.")
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
        for img_path in paths:
            # Run inference
            results = model(img_path, conf=0.10, verbose=False)[0]
            
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

            # Terminal feedback per image
            if cats_in_image > 0:
                images_with_cats += 1
                cat_detected_total += cats_in_image
                print(f"[✓] {img_path.name}: {cats_in_image} cat(s) detected.")
            else:
                print(f"[ ] {img_path.name}: No cat detected.")

            # Save the first detection as a preview image with bounding boxes
            if cats_in_image > 0 and not preview_saved:
                preview_path = EXPORTS_DIR / "preview_detection.jpg"
                results.save(filename=str(preview_path))
                print(f"  └─> Saved detection preview image to: {preview_path}")
                preview_saved = True

            # Copy image to target path
            target_img_path = ANNOTATED_DIR / split / "images" / img_path.name
            shutil.copy(img_path, target_img_path)

            # Write label text file
            target_label_path = ANNOTATED_DIR / split / "labels" / f"{img_path.stem}.txt"
            with open(target_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            processed_count += 1

    print("\n--- SUMMARY ---")
    print(f"Total images processed: {processed_count}")
    print(f"Images with cats:       {images_with_cats}")
    print(f"Total cats detected:    {cat_detected_total}")
    print(f"Dataset stored at:      {ANNOTATED_DIR}")

if __name__ == "__main__":
    main()