from torchvision import models
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=None)

        old = m.conv1
        m.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

        m.fc = nn.Identity()  # (B,512)
        self.backbone = m

    def forward(self, x):
        return self.backbone(x)