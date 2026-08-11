"""Live cat detection on RTSP stream or webcam with Home Assistant webhook notifications"""
import os
import sys
import time
import cv2
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

RTSP_URL = os.getenv("RTSP_URL", "0")
CONF_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE", "0.30"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
IMAGE_SAVE_PATH = Path(os.getenv("IMAGE_SAVE_PATH", "/mnt/usb/cat_detections"))
NOTIFICATION_COOLDOWN = int(os.getenv("NOTIFICATION_COOLDOWN", "60"))  # seconds
HA_BASE_URL = os.getenv("HA_BASE_URL", "http://homeassistant.local:8123")  # Home Assistant URL
DEVICE = get_device()

# Detect if running on Raspberry Pi
try:
    IS_RASPBERRY_PI = os.path.exists("/sys/firmware/devicetree/base/model") and \
                      "Raspberry Pi" in open("/sys/firmware/devicetree/base/model", "r").read()
except:
    IS_RASPBERRY_PI = False

def get_small_base64(image_path):
    """Compress and encode image to base64 with reduced size"""
    try:
        with Image.open(image_path) as img:
            # Reduce size
            img.thumbnail((640, 480))
            
            # Compress to buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=60)
            
            # Encode to base64
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to compress image: {e}")
        return None


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


def send_webhook_notification(image_path, confidence):
    """Send webhook notification to Home Assistant
    
    On Raspberry Pi: Sends image path/URL for Home Assistant to access locally
    On other systems: Sends compressed base64 image
    """
    if not WEBHOOK_URL:
        print("[!] Webhook disabled - no WEBHOOK_URL set")
        return False
    
    try:
        print(f"[→] Sending webhook to {WEBHOOK_URL}...")
        print(f"[→] Running on: {'Raspberry Pi' if IS_RASPBERRY_PI else 'PC/Other'}")
        
        # Prepare data
        data = {
            'message': 'Cat detected!',
            'confidence': f"{confidence:.2f}",
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'filename': os.path.basename(image_path) if image_path else None
        }
        
        # Handle image based on platform
        if image_path and os.path.exists(image_path):
            if IS_RASPBERRY_PI:
                # Raspberry Pi: Send local path/URL
                print(f"[→] Using local image path for HA: {os.path.basename(image_path)}")
                
                # Convert path to Home Assistant accessible path
                # Assumes USB stick is mounted and accessible by HA
                relative_path = image_path.replace('/mnt/usb/cat_detections/', '')
                ha_image_url = f"{HA_BASE_URL}/local/cat_detections/{relative_path}"
                
                data['image_url'] = ha_image_url
                data['local_path'] = image_path
                print(f"[✓] Image URL: {ha_image_url}")
                
            else:
                # PC/Other: Send compressed base64 image
                print(f"[→] Compressing and encoding image: {os.path.basename(image_path)}")
                
                # Get original size
                original_size = os.path.getsize(image_path)
                
                # Compress and encode
                image_base64 = get_small_base64(image_path)
                
                if image_base64:
                    compressed_size = len(image_base64)
                    data['image_base64'] = image_base64
                    print(f"[✓] Image compressed: {original_size} bytes → {compressed_size} bytes (base64)")
                else:
                    print(f"[!] Failed to compress image")
        else:
            print(f"[!] No image found at: {image_path}")
        
        # Send as JSON
        response = requests.post(WEBHOOK_URL, json=data, timeout=10)
        
        print(f"[→] Response: {response.status_code}")
        
        if response.status_code == 200:
            print(f"[✓] Webhook notification sent successfully")
            return True
        else:
            print(f"[!] Webhook failed with status {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"[ERROR] Webhook timeout after 10 seconds")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send webhook: {type(e).__name__}: {e}")
        return False


def main():
    print("=== Live Cat Detection ===")
    print(f"Model:       {MODEL_PATH.name}")
    print(f"Confidence:  {CONF_THRESHOLD}")
    print(f"Stream:      {RTSP_URL}")
    print(f"Platform:    {'Raspberry Pi 🥧' if IS_RASPBERRY_PI else 'PC/Other 💻'}")
    print(f"Webhook:     {'Enabled ✓' if WEBHOOK_URL else 'Disabled ✗'}")
    if WEBHOOK_URL:
        print(f"  URL:       {WEBHOOK_URL}")
    print(f"Images:      {IMAGE_SAVE_PATH}")
    print(f"Cooldown:    {NOTIFICATION_COOLDOWN}s")
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
                
                # Send webhook
                if send_webhook_notification(image_path, highest_conf):
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