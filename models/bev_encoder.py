import torch
import torch.nn as nn
import torchvision.models as models


class BEVEncoder(nn.Module):
    """
    Lightweight image → BEV encoder
    Outputs: [B, 16, 128, 128]
    """

    def __init__(self):
        super().__init__()

        # Backbone (ResNet18, lightweight)
        backbone = models.resnet18(weights=None)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        # Project to BEV resolution (upsample)
        self.project = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=1)
        )

        self.upsample = nn.Upsample(
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):
        """
        x: [B, 3, 256, 256]
        return: [B, 16, 128, 128]
        """

        feat = self.encoder(x)          # [B, 256, 32, 32]
        feat = self.project(feat)       # [B, 16, 32, 32]
        bev = self.upsample(feat)       # [B, 16, 128, 128]

        return bev
