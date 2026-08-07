import os
import sys
import time
import cv2
from pathlib import Path
from dotenv import load_dotenv

# Add the project root directory to sys.path so modules in src/ can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.capture.rtsp_stream import RTSPStream
from src.models.detector import CatDetector

# Load environment variables
load_dotenv(ROOT_DIR / ".env")

RTSP_URL = os.getenv("RTSP_URL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(ROOT_DIR / "data" / "raw"))
INTERVAL_SECONDS = 0.5
CONFIDENCE_THRESHOLD = 0.1

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing RTSP Stream and Detector...")
    stream = RTSPStream(RTSP_URL)
    detector = CatDetector(model_name="yolov8m.pt", conf_threshold=CONFIDENCE_THRESHOLD)

    last_saved_time = time.time()
    img_count = 0

    print("Data collection started. Saving images with a cat ONLY when no human is detected...")
    print("Press 'q' in the preview window to quit.")

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
            cv2.imshow("Tapo C310 - Live Cat Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting stream on keypress.")
                break

    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()