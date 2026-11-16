import torch
import torch.nn as nn


class TemporalEncoderV2(nn.Module):
    """
    Temporal Encoder supporting both:
    1. Object-Centric Trajectory Encoding (default DOCOF):
       Encodes historical object state sequences [x, y, vx, vy, phi, w, l] using a GRU.
       Inputs:
         - history: [B, T, N, state_dim] or [B, N, T, state_dim]
       Outputs:
         - h_t: [B, N, hidden_dim]

    2. Dense BEV Temporal Encoding (for baseline comparison):
       Encodes multi-frame BEV feature grids [B, T, C, H, W] via 3D/spatio-temporal convolutions.
       Outputs:
         - tokens, H, W
    """

    def __init__(self, state_dim=7, hidden_dim=128, in_channels=None, d_model=None):
        super().__init__()

        # Handle keyword differences across invocations
        if in_channels is not None or d_model is not None:
            self.mode = "dense_bev"
            in_c = in_channels if in_channels is not None else 1
            dim = d_model if d_model is not None else hidden_dim
            self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
            self.temporal_conv = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1)
            self.norm = nn.BatchNorm2d(dim)
            self.relu = nn.ReLU(inplace=True)
            self.d_model = dim
        else:
            self.mode = "object_centric"
            self.state_dim = state_dim
            self.hidden_dim = hidden_dim
            self.gru = nn.GRU(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            )

    def forward(self, x):
        if self.mode == "dense_bev" or (x.dim() == 5):
            # Dense BEV sequence: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            feats = self.relu(self.norm(self.conv_in(x_flat)))
            feats = feats.view(B, T, self.d_model, H, W).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            fused = self.temporal_conv(feats)[:, :, -1]  # Take last time step: [B, C, H, W]
            tokens = fused.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            return tokens, H, W

        # Object-centric: [B, T, N, state_dim] or [B, N, T, state_dim]
        if x.dim() == 4:
            if x.shape[1] == 3 and x.shape[2] != 3:  # Shape is (B, T, N, D) with T=3
                B, T, N, D = x.shape
                x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
            else:  # Shape is (B, N, T, D)
                B, N, T, D = x.shape
                x = x.contiguous().view(B * N, T, D)
            _, h_n = self.gru(x)  # h_n: [1, B*N, hidden_dim]
            h_t = h_n.squeeze(0).view(B, N, self.hidden_dim)
            return h_t

        elif x.dim() == 3:  # [N, T, D]
            N, T, D = x.shape
            _, h_n = self.gru(x)
            return h_n.squeeze(0).view(1, N, self.hidden_dim)

        else:
            raise ValueError(f"Unsupported input dimension for TemporalEncoder: {x.shape}")
