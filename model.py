import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class YOLOModel(nn.Module):
    def __init__(self, num_classes=20, grid_size=7):
        super(YOLOModel, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

        # Load pretrained MobileNetV2 backbone features with updated weights parameter
        self.backbone = mobilenet_v2(weights=mobilenet_v2.MobileNet_V2_Weights.DEFAULT).features
        self.backbone_out_channels = 1280  # Output channels from MobileNetV2 features

        # Detection head: outputs (5 + num_classes) channels per grid cell
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 5 + num_classes, kernel_size=1)  # (objectness + 4 bbox coords + num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.backbone(x)  # Shape: (batch, 1280, H, W)
        out = self.head(features)    # Shape: (batch, 5+num_classes, H, W)

        # Resize output to (batch, grid_size, grid_size, 5+num_classes)
        out = nn.functional.interpolate(out, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
        out = out.permute(0, 2, 3, 1).contiguous()  # (batch, grid_size, grid_size, 5 + num_classes)
        return out
