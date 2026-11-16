import torch
import torch.nn as nn


class TemporalBackbone(nn.Module):
    """
    ConvGRU temporal fusion.
    Input:  x -> [B, T, C, H, W]
    Output: fused BEV -> [B, C, H, W]
    """

    def __init__(self, channels=16):
        super().__init__()

        self.channels = channels

        self.conv_z = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=3,
            padding=1
        )

        self.conv_r = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=3,
            padding=1
        )

        self.conv_h = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """

        B, T, C, H, W = x.shape

        # initial hidden state
        h = x[:, 0]  # [B, C, H, W]

        for t in range(1, T):
            current = x[:, t]

            combined = torch.cat([h, current], dim=1)

            z = self.sigmoid(self.conv_z(combined))
            r = self.sigmoid(self.conv_r(combined))

            combined_r = torch.cat([r * h, current], dim=1)
            h_tilde = self.tanh(self.conv_h(combined_r))

            h = (1 - z) * h + z * h_tilde

        return h
