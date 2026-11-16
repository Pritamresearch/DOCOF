import torch.nn as nn
from models.temporal_encoder_v2 import TemporalEncoderV2
from models.motion_model_v2 import MotionModelV2
from models.renderer_v2 import DifferentiableRendererV2


class DOCOFv2(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.encoder = TemporalEncoderV2(
            cfg["state_dim"],
            cfg["hidden_dim"]
        )

        self.motion = MotionModelV2(
            hidden_dim=cfg["hidden_dim"],
            state_dim=cfg["state_dim"],
            future_frames=cfg["future_frames"],
            dt=cfg["dt"],
            max_speed=cfg["max_speed"],
            max_acc=cfg["max_acc"],
            max_omega=cfg["max_omega"]
        )

        self.renderer = DifferentiableRendererV2(
            grid_size=cfg["grid_size"],
            bev_range=cfg["bev_range"]
        )

    def forward(self, history, state_t, validity_masks):

        h_t = self.encoder(history)

        future_states, controls = self.motion(
            state_t,
            h_t,
            validity_masks
        )

        occupancies = []
        for k in range(len(future_states)):
            occ = self.renderer(
                future_states[k],
                validity_masks[k]
            )
            occupancies.append(occ)

        return future_states, occupancies, controls