"""Live cat detection on RTSP stream with Telegram notifications"""
import os
import sys
import time
import cv2
import numpy as np
import requests
import base64
import io
from pathlib import Path
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# Force UTF-8 encoding for Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Python < 3.7

# Add project root to path for imports
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

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

RTSP_URL = os.getenv("RTSP_URL")
if not RTSP_URL:
    raise ValueError("RTSP_URL must be set in .env file")
CONF_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE", "0.30"))
IMAGE_SAVE_PATH = Path(os.getenv("IMAGE_SAVE_PATH", "/mnt/usb/cat_detections"))
NOTIFICATION_COOLDOWN = int(os.getenv("NOTIFICATION_COOLDOWN", "60"))  # seconds

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DEVICE = get_device()

def save_detection_image(frame, boxes, confidence):
    """Save detection image to USB stick and return the file path"""
    try:
        IMAGE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cat_{timestamp}_conf{confidence:.2f}.jpg"
        image_path = IMAGE_SAVE_PATH / filename
        
        # Save annotated frame
        success = cv2.imwrite(str(image_path), frame)
        
        if success:
            print(f"[✓] Image saved: {filename}")
            return str(image_path)
        else:
            print(f"[ERROR] Failed to write image to {image_path}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to save image: {type(e).__name__}: {e}")
        return None


def send_telegram_notification(image_path, confidence):
    """Send notification via Telegram with image"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram disabled - TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return False
    
    try:
        print(f"[→] Sending Telegram notification...")
        
        # Send photo with caption
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        caption = f"🐱 Fussel detected!\nConfidence: {confidence:.2f}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': caption
            }
            response = requests.post(url, files=files, data=data, timeout=10)
        
        if response.status_code == 200:
            print(f"[✓] Telegram notification sent successfully")
            return True
        else:
            print(f"[!] Telegram failed with status {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram: {type(e).__name__}: {e}")
        return False


def debug_test():
    """Test notification with a fake detection"""
    print("\n=== 🐛 DEBUG MODE: Notification Test ===")
    print(f"Telegram:    {TELEGRAM_BOT_TOKEN[:20]}..." if TELEGRAM_BOT_TOKEN else "Not configured")
    print(f"Images:      {IMAGE_SAVE_PATH}\n")
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return
    
    # Create a test image
    print("[→] Creating test image...")
    test_frame = cv2.imread(str(PROJECT_ROOT / "data" / "raw" / "cats" / "cat_001.jpg"))
    
    if test_frame is None:
        # Create a dummy image if no test image exists
        print("[!] No test image found, creating dummy image...")
        test_frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "TEST CAT DETECTION", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.rectangle(test_frame, (200, 150), (440, 350), (0, 255, 0), 3)
    
            
    # Save test image
    IMAGE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_image_path = IMAGE_SAVE_PATH / f"test_{timestamp}_conf0.95.jpg"
    cv2.imwrite(str(test_image_path), test_frame)
    print(f"[✓] Test image created: {test_image_path.name}")
    
        # Send test notification
    print("\n[→] Sending test notification...")
    test_confidence = 0.95
    
    success = send_telegram_notification(str(test_image_path), test_confidence)
    
    if success:
        print("\n[✓] ✅ DEBUG TEST SUCCESSFUL!")
        print("    Check your Telegram app for the notification.")
    else:
        print("\n[✗] ❌ DEBUG TEST FAILED!")
        print("    Check the error messages above.")
    
    print(f"\n[i] Test image saved at: {test_image_path}")


def main(debug=False):
    print("=== Live Cat Detection ===")
    print(f"Model:       {MODEL_PATH.name}")
    print(f"Confidence:  {CONF_THRESHOLD}")
    print(f"Stream:      {RTSP_URL}")
    print(f"Debug Mode:  {'🐛 ENABLED' if debug else 'Disabled'}")
    print(f"Telegram:    {'Enabled ✓' if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else 'Disabled ✗'}")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print(f"  Chat ID:   {TELEGRAM_CHAT_ID}")
    print(f"Images:      {IMAGE_SAVE_PATH}")
    print(f"Cooldown:    {NOTIFICATION_COOLDOWN}s")
    print_device_info()
    
        # Run debug mode if enabled
    if debug:
        debug_test()
        return
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print(f"        Please train the model first: python main.py train")
        return
    
    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    print(f"Connecting to RTSP stream...")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {RTSP_URL}")
        return

    print("\n[✓] Live detector running. Press 'q' to quit.\n")

    cat_detected_prev = None
    last_notification_time = 0

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

        # Generate annotated frame with bounding boxes (needed before saving)
        annotated_frame = results.plot()

                # Handle detections
        if cat_detected:
            conf_list = [b.conf.item() for b in results.boxes]
            highest_conf = max(conf_list) if conf_list else 0.0
            
            # Console output on state change
            if cat_detected != cat_detected_prev:
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] 🐱 CAT DETECTED! (Confidence: {highest_conf:.2f})")
            
            # Send notification with cooldown (continuous while cat is detected)
            current_time = time.time()
            time_since_last = current_time - last_notification_time
            
            if time_since_last >= NOTIFICATION_COOLDOWN:
                print(f"[→] Cooldown passed ({time_since_last:.0f}s), sending notification...")
                # Save image
                image_path = save_detection_image(annotated_frame, results.boxes, highest_conf)
                
                                # Send Telegram notification
                if send_telegram_notification(image_path, highest_conf):
                    last_notification_time = current_time
            # Removed cooldown message to reduce spam
                
        else:
            # No cat detected - log state change only
            if cat_detected != cat_detected_prev:
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] ❌ No cat detected")
        
        cat_detected_prev = cat_detected

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

        # Display window (only if not in headless mode)
        try:
            cv2.imshow("Smart Home Cat Detector", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # Headless mode - no display available
            time.sleep(0.01)  # Small delay to prevent CPU overload
            
    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] Stream stopped.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        cv2.destroyAllWindows()