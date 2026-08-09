import os
import sys
import time
import cv2
import torch
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# --- Project Root Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ultralytics import YOLO

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# --- Find Model Weights ---
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "cat_yolov8n.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "cat_yolov8n.pt"

if not MODEL_PATH.exists():
    print(f"[Error] Model weights not found at: {MODEL_PATH}")
    sys.exit(1)

# RTSP Stream URL from .env or 0 for local Webcam
RTSP_URL = os.getenv("RTSP_URL", 0)

# Lower threshold to 0.30 so cats are detected more easily
CONF_THRESHOLD = 0.30
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def main():
    print(f"Loading YOLO model: {MODEL_PATH.name} on device '{DEVICE}'...")
    model = YOLO(str(MODEL_PATH))

    print(f"Connecting to stream: {RTSP_URL} ...")
    cap = cv2.VideoCapture(int(RTSP_URL) if str(RTSP_URL).isdigit() else str(RTSP_URL))

    if not cap.isOpened():
        print("[Error] Could not open video stream.")
        sys.exit(1)

    print("\n[OK] Live Detector running. Press 'q' in the window to quit.\n")

    cat_detected_prev = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame. Reconnecting...")
            time.sleep(1)
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

        # --- Console Output (State changes only) ---
        if cat_detected != cat_detected_prev:
            timestamp = time.strftime('%H:%M:%S')
            if cat_detected:
                # FIXED: Correct variable iteration to fix NameError
                conf_list = [b.conf.item() for b in results.boxes]
                highest_conf = max(conf_list) if conf_list else 0.0
                print(f"[{timestamp}] STATUS: 🐱 Fussel Detected! (Conf: {highest_conf:.2f})")
            else:
                print(f"[{timestamp}] STATUS: ❌ No Fussel Detected")
            cat_detected_prev = cat_detected

        # First generate bounding box plot
        annotated_frame = results.plot()

        # --- Top-Right Status Overlay (English) ---
        text_str = "STATUS: Fussel DETECTED" if cat_detected else "STATUS: NO Fussel"
        text_color = (0, 255, 0) if cat_detected else (0, 0, 255)  # Green vs Red

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
    print("Stream stopped.")

if __name__ == "__main__":
    main()