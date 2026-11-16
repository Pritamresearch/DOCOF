import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionModelV2(nn.Module):

    def __init__(self,
                 hidden_dim,
                 state_dim,
                 future_frames,
                 dt,
                 max_speed,
                 max_acc,
                 max_omega):

        super().__init__()

        self.future_frames = future_frames
        self.dt = dt
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.max_omega = max_omega

        self.gru_cell = nn.GRUCell(
            hidden_dim + 3,  # horizon embedding + hidden
            hidden_dim
        )

        self.control_head = nn.Linear(
            hidden_dim,
            3  # ax, ay, omega
        )

        self.horizon_embed = nn.Embedding(
            future_frames,
            3
        )

    def forward(self, state_t, hidden, validity_masks):
        # print("future_frames:", self.future_frames)  
        # print("validity_masks len:", len(validity_masks))  
        B, N, _ = state_t.shape

        states = []
        controls = []

        current_state = state_t
        hidden_state = hidden

        for k in range(self.future_frames):

            # ----------------------------------
            # Horizon embedding
            # ----------------------------------

            horizon_ids = torch.full(
                (B * N,),
                k,
                device=current_state.device,
                dtype=torch.long
            )

            horizon_vec = self.horizon_embed(horizon_ids)
            horizon_vec = horizon_vec.view(B, N, -1)

            # ----------------------------------
            # GRU update
            # ----------------------------------

            gru_input = torch.cat(
                [hidden_state, horizon_vec],
                dim=-1
            )

            hidden_flat = hidden_state.reshape(B * N, -1)
            input_flat = gru_input.reshape(B * N, -1)

            hidden_next = self.gru_cell(
                input_flat,
                hidden_flat
            )

            hidden_next = hidden_next.view(B, N, -1)

            # Freeze invalid objects numerically
            valid_k = validity_masks[k].unsqueeze(-1).bool()
            hidden_state = torch.where(valid_k, hidden_next, hidden_state)

            # ----------------------------------
            # Predict control
            # ----------------------------------

            control = self.control_head(hidden_state)

            # Clamp controls safely (NO inplace)
            ax = torch.clamp(control[..., 0], -self.max_acc, self.max_acc)
            ay = torch.clamp(control[..., 1], -self.max_acc, self.max_acc)
            omega = torch.clamp(control[..., 2], -self.max_omega, self.max_omega)

            control = torch.stack([ax, ay, omega], dim=-1)
            controls.append(control)

            # ----------------------------------
            # Structured update
            # ----------------------------------

            x, y, vx, vy, phi, w, l = torch.unbind(current_state, dim=-1)

            # velocity update
            vx_next = vx + ax * self.dt
            vy_next = vy + ay * self.dt

            # speed clamp (SAFE)
            speed = torch.sqrt(vx_next * vx_next + vy_next * vy_next + 1e-6)
            scale = (self.max_speed / (speed + 1e-6)).clamp(max=1.0)
            vx_next = vx_next * scale
            vy_next = vy_next * scale

            # position update
            x_next = x + vx_next * self.dt
            y_next = y + vy_next * self.dt

            # yaw update
            phi_next = phi + omega * self.dt

            next_state = torch.stack(
                [x_next, y_next, vx_next, vy_next, phi_next, w, l],
                dim=-1
            )

            # Freeze invalid objects numerically
            valid_k = validity_masks[k].unsqueeze(-1).bool()
            current_state = torch.where(valid_k, next_state, current_state)

            states.append(current_state)

        return states, controls