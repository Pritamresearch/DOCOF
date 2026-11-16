import torch
import torch.nn as nn


class ForecastDecoderV2(nn.Module):
    """
    Decodes conditioned BEV tokens to occupancy map.
    """
    def __init__(self, d_model=128, future_frames=3):
        super().__init__()

        self.future_frames = future_frames
        self.conv = nn.Conv2d(d_model, future_frames, kernel_size=1)

        # self.conv = nn.Sequential(
        # nn.Conv2d(d_model, 64, 3, padding=1),
        # nn.ReLU(),
        # nn.Conv2d(64, 1, 1)
        #     )

    def forward(self, tokens, H, W):

        B, HW, D = tokens.shape

        feature_map = tokens.permute(0, 2, 1).view(B, D, H, W)

        out = self.conv(feature_map)  # [B, K, H, W]

        return out.unsqueeze(2)  # [B, K, 1, H, W]