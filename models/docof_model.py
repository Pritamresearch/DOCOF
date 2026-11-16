# models/docof_model.py

import torch
import torch.nn as nn


class DOCOF(nn.Module):
    def __init__(self, grid_size, num_classes, future_frames=4):
        super().__init__()

        self.future_frames = future_frames
        self.num_classes = num_classes

        # -------------------------
        # Spatial Encoder
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        # -------------------------
        # Temporal Fusion (3D Conv)
        # Input shape: (B, C, T, H, W)
        # -------------------------
        self.temporal_fusion = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3, 3),
            padding=1
        )

        # -------------------------
        # Multi-horizon Head
        # Output channels = C * K
        # -------------------------
        self.head = nn.Conv2d(
            128,
            num_classes * future_frames,
            kernel_size=1
        )

    def forward(self, past_bev):

        # past_bev: (B, T, 1, H, W)
        B, T, C, H, W = past_bev.shape

        spatial_features = []

        for t in range(T):
            f = self.encoder(past_bev[:, t])  # (B,128,H,W)
            spatial_features.append(f)

        # (B,128,T,H,W)
        features = torch.stack(spatial_features, dim=2)

        # 3D temporal fusion
        fused = self.temporal_fusion(features)

        # Use last temporal slice
        fused = fused[:, :, -1]  # (B,128,H,W)

        # Multi-horizon prediction
        out = self.head(fused)  # (B, C*K, H, W)

        # Reshape to (B, K, C, H, W)
        out = out.view(
            B,
            self.future_frames,
            self.num_classes,
            H,
            W
        )

        return out