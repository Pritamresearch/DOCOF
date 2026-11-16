import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_z = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.conv_r = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.conv_h = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x, h):

        combined = torch.cat([x, h], dim=1)

        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))

        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))

        h_new = (1 - z) * h + z * h_tilde

        return h_new


class ObjectTemporalModel(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=96, num_classes=3):
        super().__init__()

        # Stronger spatial encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # Temporal unit
        self.gru = ConvGRUCell(hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

    def forward(self, x):
        # x: [B,T,3,H,W]

        B, T, C, H, W = x.shape

        h = torch.zeros(B, 96, H, W, device=x.device)

        for t in range(T):
            xt = self.encoder(x[:, t])
            h = self.gru(xt, h)

        out = self.decoder(h)

        return out
