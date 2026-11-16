import torch
import torch.nn as nn


class DifferentiableRendererV2(nn.Module):

    def __init__(self, grid_size=200, bev_range=50):
        super().__init__()

        coords = torch.linspace(-bev_range, bev_range, grid_size)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="xy")

        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def forward(self, states, validity_mask):

        B, N, _ = states.shape

        x = states[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        y = states[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        yaw = states[:, :, 4].unsqueeze(-1).unsqueeze(-1)
        w = states[:, :, 5].unsqueeze(-1).unsqueeze(-1)
        l = states[:, :, 6].unsqueeze(-1).unsqueeze(-1)

        dx = self.grid_x.unsqueeze(0).unsqueeze(0) - x
        dy = self.grid_y.unsqueeze(0).unsqueeze(0) - y

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        x_rot = cos_y * dx + sin_y * dy
        y_rot = -sin_y * dx + cos_y * dy

        box_prob = torch.sigmoid(-(torch.abs(x_rot) - w/2)*6) * \
                   torch.sigmoid(-(torch.abs(y_rot) - l/2)*6)

        valid = validity_mask.unsqueeze(-1).unsqueeze(-1)
        box_prob *= valid

        eps = 1e-6
        log_union = torch.sum(
            torch.log1p(-box_prob + eps),
            dim=1
        )

        eps = 1e-6
        log_union = torch.sum(
            torch.log1p(-box_prob + eps),
            dim=1
        )

        prob = 1 - torch.exp(log_union)
        prob = torch.clamp(prob, 1e-6, 1-1e-6)

        return prob