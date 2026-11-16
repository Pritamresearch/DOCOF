import torch
import torch.nn as nn


class OccupancyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, 1, H, W]
        """
        return self.head(x)
