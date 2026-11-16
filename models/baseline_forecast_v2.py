import torch
import torch.nn as nn

from models.temporal_encoder_v2 import TemporalEncoderV2
from models.forecast_decoder_v2 import ForecastDecoderV2


class BaselineForecastV2(nn.Module):

    def __init__(self, d_model=128):
        super().__init__()

        self.temporal_encoder = TemporalEncoderV2(
            in_channels=1,
            d_model=d_model
        )

        # 🔴 Force multi-frame decoder explicitly
        self.decoder = ForecastDecoderV2(
            d_model=d_model,
            future_frames=3
        )

    def forward(self, bev_sequence):

        bev_tokens, H, W = self.temporal_encoder(bev_sequence)

        future_bev = self.decoder(bev_tokens, H, W)

        return future_bev