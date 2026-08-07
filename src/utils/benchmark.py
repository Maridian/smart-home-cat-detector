import time
import os
import torch
import cv2
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F

TEST_IMAGE_PATH = r"data\test_sample.jpg"  # Pfad zu einem Testbild
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. YOLOv8m Benchmark
print("=== Benchmark: YOLOv8m ===")
yolo_model = YOLO("yolov8m.pt").to(DEVICE)
frame = cv2.imread(TEST_IMAGE_PATH)

# Warmup
for _ in range(5):
    _ = yolo_model(frame, verbose=False)

start_time = time.time()
num_runs = 100
for _ in range(num_runs):
    _ = yolo_model(frame, verbose=False)
yolo_latency = ((time.time() - start_time) / num_runs) * 1000

# 2. PyTorch SSDLite320 Benchmark
print("=== Benchmark: PyTorch SSDLite320 ===")
pytorch_model = ssdlite320_mobilenet_v3_large(num_classes=2)
if os.path.exists("cat_ssdlite320.pth"):
    pytorch_model.load_state_dict(torch.load("cat_ssdlite320.pth", map_location=DEVICE))
pytorch_model.to(DEVICE).eval()

img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(DEVICE)

# Warmup
with torch.no_grad():
    for _ in range(5):
        _ = pytorch_model(img_tensor)

start_time = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = pytorch_model(img_tensor)
pytorch_latency = ((time.time() - start_time) / num_runs) * 1000

# Ergebnisse ausgeben
print("\n" + "="*40)
print(f"Hardware: {DEVICE}")
print(f"YOLOv8m Latenz:        {yolo_latency:.2f} ms / Bild ({1000/yolo_latency:.1f} FPS)")
print(f"SSDLite320 Latenz:     {pytorch_latency:.2f} ms / Bild ({1000/pytorch_latency:.1f} FPS)")
print("="*40)