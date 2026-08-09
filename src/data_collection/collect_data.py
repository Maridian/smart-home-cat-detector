"""Data collection script for gathering cat images from RTSP stream"""
import os
import time
import cv2

from src.utils.config import setup_project_path, get_project_root, load_env_config
from src.capture.rtsp_stream import RTSPStream
from src.models.detector import CatDetector

# Setup paths
setup_project_path()
load_env_config()

# Configuration
RTSP_URL = os.getenv("RTSP_URL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(get_project_root() / "data" / "raw"))
INTERVAL_SECONDS = float(os.getenv("SAVE_INTERVAL", "0.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.1"))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not RTSP_URL:
        print("[ERROR] RTSP_URL not set in .env file")
        return
    
    print("Initializing RTSP Stream and Detector...")
    print(f"  Stream URL: {RTSP_URL}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Save Interval: {INTERVAL_SECONDS}s")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    stream = RTSPStream(RTSP_URL)
    detector = CatDetector(model_name="yolov8m.pt", conf_threshold=CONFIDENCE_THRESHOLD)

    last_saved_time = time.time()
    img_count = 0

    print("\nData collection started.")
    print("Saving images with cat ONLY when no human is detected...")
    print("Press 'q' in the preview window to quit.\n")

    try:
        while True:
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                continue

            # Run inference using the detector module
            should_save, conf, annotated_frame = detector.process_frame(frame)

            current_time = time.time()
            if should_save and (current_time - last_saved_time >= INTERVAL_SECONDS):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(OUTPUT_DIR, f"cat_{timestamp}.jpg")

                # Save raw unannotated image for training dataset
                cv2.imwrite(filename, frame)
                img_count += 1
                print(f"[{img_count}] Cat detected without human ({conf:.1f}%)! Saved: {filename}")
                last_saved_time = current_time

            # Show live stream preview
            cv2.imshow("Cat Detector - Data Collection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting stream on keypress.")
                break

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print(f"\n[✓] Collection complete. Total images saved: {img_count}")

if __name__ == "__main__":
    main()