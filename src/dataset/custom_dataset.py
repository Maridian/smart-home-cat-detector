import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_c, y_c, bw, bh = map(float, parts)
                        # Umrechnung von normiertem YOLO-Format in absolute PyTorch Pixel-Koordinaten [x1, y1, x2, y2]
                        x1 = (x_c - bw / 2) * w
                        y1 = (y_c - bh / 2) * h
                        x2 = (x_c + bw / 2) * w
                        y2 = (y_c + bh / 2) * h
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)  # In torchvision ist Class 0 = Hintergrund, Class 1 = Katze

        # Falls keine Box vorhanden ist (z.B. Negativ-Beispiel)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        # Konvertierung in Tensor (Skalierung 0-1)
        image_tensor = F.to_tensor(image)

        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))