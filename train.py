model_code = """
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class YOLOModel(nn.Module):
    def __init__(self, num_classes=20):
        super(YOLOModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = mobilenet_v2(pretrained=True).features
        self.backbone_out_channels = 1280

        self.head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, (5 + num_classes), kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.backbone(x)  # (batch, 1280, H, W)
        out = self.head(features)    # (batch, 5+num_classes, H, W)
        out = out.permute(0, 2, 3, 1).contiguous()  # (batch, H, W, 5+num_classes)
        return out
"""

with open("model.py", "w") as f:
    f.write(model_code)
print(" model.py saved!")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
from model import YOLOModel
from loss import YoloLoss
import os
import numpy as np

# Grid and classes as before
GRID_SIZE = 7
NUM_CLASSES = 20  # Pascal VOC has 20 classes
IMG_SIZE = 224

# Map VOC class names to index
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}

# Transform to resize and normalize images for MobileNetV2
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def encode_label(target, grid_size=GRID_SIZE, num_classes=NUM_CLASSES):
    """
    Convert VOC target (dict) to YOLO label tensor of shape (grid_size, grid_size, 5 + num_classes).
    Format per cell:
    [objectness, bbox_x, bbox_y, bbox_w, bbox_h, class_one_hot...]

    bbox coordinates normalized to [0,1] relative to image, and relative to grid cell.
    """
    label = torch.zeros(grid_size, grid_size, 5 + num_classes)

    # VOC target['annotation']['object'] can be a list or dict if one object
    objs = target['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]

    for obj in objs:
        cls_name = obj['name']
        if cls_name not in class_to_idx:
            continue  # skip unknown classes
        cls_idx = class_to_idx[cls_name]

        bbox = obj['bndbox']
        # Original VOC bbox is in absolute pixel values (xmin, ymin, xmax, ymax)
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])

        # VOC images vary in size; normalize bbox coords relative to original image size
        img_w = float(target['annotation']['size']['width'])
        img_h = float(target['annotation']['size']['height'])

        # normalize bbox coords between 0-1
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Determine grid cell containing center
        cell_x = int(x_center * grid_size)
        cell_y = int(y_center * grid_size)
        if cell_x >= grid_size:
            cell_x = grid_size - 1
        if cell_y >= grid_size:
            cell_y = grid_size - 1

        # Position relative to cell
        x_cell = x_center * grid_size - cell_x
        y_cell = y_center * grid_size - cell_y

        # Fill label tensor at that grid cell
        label[cell_y, cell_x, 0] = 1.0  # objectness
        label[cell_y, cell_x, 1] = x_cell
        label[cell_y, cell_x, 2] = y_cell
        label[cell_y, cell_x, 3] = w
        label[cell_y, cell_x, 4] = h
        label[cell_y, cell_x, 5 + cls_idx] = 1.0

    return label

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root='./VOCdevkit/', year='2012', image_set='train', transform=None):
        self.dataset = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)

        label = encode_label(target)
        return img, label

def train(epochs=20, batch_size=8, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    dataset = VOCDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = YOLOModel(num_classes=NUM_CLASSES).to(device)
    criterion = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/yolo_model.pth')
    print("Training complete. Model saved to weights/yolo_model.pth")

if __name__ == "__main__":
    train()



