"""Live cat detection on RTSP stream or webcam"""
import os
import sys
import time
import cv2
from pathlib import Path
from ultralytics import YOLO

# Force UTF-8 encoding for Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Python < 3.7

from src.utils.config import setup_project_path, get_project_root, load_env_config
from src.utils.device import get_device, print_device_info

# Setup
setup_project_path()
load_env_config()

# Configuration
PROJECT_ROOT = get_project_root()
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "cat_yolov8n.pt"

if not MODEL_PATH.exists():
    # Fallback to root directory
    MODEL_PATH = PROJECT_ROOT / "cat_yolov8n.pt"

RTSP_URL = os.getenv("RTSP_URL", "0")
CONF_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE", "0.30"))
DEVICE = get_device()

def main():
    print("=== Live Cat Detection ===")
    print(f"Model:      {MODEL_PATH.name}")
    print(f"Confidence: {CONF_THRESHOLD}")
    print(f"Stream:     {RTSP_URL}")
    print_device_info()
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print(f"        Please train the model first: python main.py train")
        return
    
    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    print(f"Connecting to stream...")
    # Handle webcam (0) or RTSP URL
    stream_source = int(RTSP_URL) if RTSP_URL.isdigit() else RTSP_URL
    cap = cv2.VideoCapture(stream_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {RTSP_URL}")
        return

    print("\n[✓] Live detector running. Press 'q' to quit.\n")

    cat_detected_prev = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame. Reconnecting...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(stream_source)
            continue

        # Run inference
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )[0]

        # Check for detections
        cat_detected = len(results.boxes) > 0

                # Console output on state changes only
        if cat_detected != cat_detected_prev:
            timestamp = time.strftime('%H:%M:%S')
            if cat_detected:
                conf_list = [b.conf.item() for b in results.boxes]
                highest_conf = max(conf_list) if conf_list else 0.0
                print(f"[{timestamp}] 🐱 CAT DETECTED! (Confidence: {highest_conf:.2f})")
            else:
                print(f"[{timestamp}] ❌ No cat detected")
            cat_detected_prev = cat_detected

                # Generate annotated frame with bounding boxes
        annotated_frame = results.plot()

        # Status overlay (top-right)
        text_str = "CAT DETECTED" if cat_detected else "NO CAT"
        text_color = (0, 255, 0) if cat_detected else (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2

        (w, h), baseline = cv2.getTextSize(text_str, font, scale, thick)
        img_h, img_w = annotated_frame.shape[:2]

        margin_right = 20
        margin_top = 35
        x = img_w - w - margin_right
        y = margin_top

        pad = 6
        cv2.rectangle(
            annotated_frame,
            (x - pad, y - h - pad),
            (x + w + pad, y + baseline + pad),
            (0, 0, 0),
            cv2.FILLED
        )

        cv2.putText(
            annotated_frame,
            text_str,
            (x, y),
            font,
            scale,
            text_color,
            thick,
            cv2.LINE_AA
        )

        # Display window
        cv2.imshow("Smart Home Cat Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] Stream stopped.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        cv2.destroyAllWindows()