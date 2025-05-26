import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, grid_size=7, num_classes=20):
        super(YoloLoss, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predictions, target):
        # obj_mask shape: (batch, grid_size, grid_size)
        obj_mask = target[..., 0]
        noobj_mask = 1 - obj_mask

        # Objectness loss (where object exists)
        obj_loss = self.mse(obj_mask * predictions[..., 0], obj_mask * target[..., 0])
        # No objectness loss (where no object)
        noobj_loss = self.mse(noobj_mask * predictions[..., 0], noobj_mask * target[..., 0])

        # Coordinate loss (only for cells with objects)
        coord_loss = self.mse(
            obj_mask.unsqueeze(-1) * predictions[..., 1:5],
            obj_mask.unsqueeze(-1) * target[..., 1:5]
        )

        # Class prediction loss (only for cells with objects)
        class_loss = self.mse(
            obj_mask.unsqueeze(-1) * predictions[..., 5:],
            obj_mask.unsqueeze(-1) * target[..., 5:]
        )

        total_loss = obj_loss + 0.5 * noobj_loss + coord_loss + class_loss
        return total_loss
