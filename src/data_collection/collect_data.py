"""Data collection script for gathering cat images from RTSP stream"""
import os
import sys
import time
import cv2
import argparse

from src.utils.config import setup_project_path, get_project_root, load_env_config
from src.utils.rtsp_stream import RTSPStream
from src.models.detector import CatDetector

# Setup paths
setup_project_path()
load_env_config()

# Configuration
RTSP_URL = os.getenv("RTSP_URL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(get_project_root() / "data" / "raw"))
INTERVAL_SECONDS = float(os.getenv("SAVE_INTERVAL", "0.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.1"))

def collect_with_detection():
    """Collect images only when cat is detected"""
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
    print("Saving images whenever a cat is detected...")
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
                print(f"[{img_count}] Cat detected ({conf:.1f}%)! Saved: {filename}")
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

def collect_force():
    """Collect images WITHOUT detection - save every frame (for negative samples)"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not RTSP_URL:
        print("[ERROR] RTSP_URL not set in .env file")
        return
    
    print("\n=== FORCE MODE: Collecting ALL frames (no detection) ===")
    print(f"  Stream URL: {RTSP_URL}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Save Interval: {INTERVAL_SECONDS}s")
    print("\n  This mode saves frames WITHOUT running detection.")
    print("  Perfect for collecting negative samples (no cats).\n")
    
    stream = RTSPStream(RTSP_URL)

    last_saved_time = time.time()
    img_count = 0

    print("Data collection started (FORCE MODE).")
    print("Saving every frame at specified interval...")
    print("Press 'q' in the preview window to quit.\n")

    try:
        while True:
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                continue

            current_time = time.time()
            
            # Save frame at interval (no detection needed)
            if current_time - last_saved_time >= INTERVAL_SECONDS:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(OUTPUT_DIR, f"cat_{timestamp}.jpg")

                cv2.imwrite(filename, frame)
                img_count += 1
                print(f"[{img_count}] Frame saved: {filename}")
                last_saved_time = current_time

            # Show live stream preview (without annotations)
            # Skip display on headless systems
            if os.getenv("DISPLAY") and not os.getenv("HEADLESS_MODE"):
                try:
                    cv2.imshow("Data Collection - FORCE MODE (No Detection)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting stream on keypress.")
                        break
                except cv2.error:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print(f"\n[✓] Collection complete. Total images saved: {img_count}")


def main(force=False):
    """Main entry point"""
    if force:
        collect_force()
    else:
        collect_with_detection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect training data from RTSP camera stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data_collection.collect_data              # Normal mode: save when cat detected
  python -m src.data_collection.collect_data --force      # Force mode: save all frames
  
  python main.py collect                                  # Via main.py (normal mode)
  python main.py collect --force                          # Via main.py (force mode)
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force mode: Save all frames without detection (for negative samples)'
    )
    
    args = parser.parse_args()
    
    try:
        main(force=args.force)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(0)