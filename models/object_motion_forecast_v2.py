import torch
import torch.nn as nn
import torchvision.ops as ops


class ObjectMotionForecastV2(nn.Module):

    def __init__(self, future_frames=3, num_intent_classes=5, roi_size=7):
        super().__init__()

        self.future_frames = future_frames
        self.roi_size = roi_size

        # Temporal Encoder (stronger)
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # Collapse temporal dimension
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

        # Motion head
        self.motion_mlp = nn.Sequential(
            nn.Linear(64 * roi_size * roi_size, 256),
            nn.ReLU(),
            nn.Linear(256, future_frames * 2)
        )

        # Intent head
        self.intent_mlp = nn.Sequential(
            nn.Linear(64 * roi_size * roi_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_intent_classes)
        )

    def forward(self, bev_past, boxes):

        B = bev_past.size(0)

        # [B, T, 1, H, W] → [B, 1, T, H, W]
        bev_past = bev_past.permute(0, 2, 1, 3, 4)

        feat = self.temporal_encoder(bev_past)
        feat = self.temporal_pool(feat).squeeze(2)  # [B, C, H, W]

        intent_preds = []
        motion_preds = []

        for b in range(B):

            if boxes[b].numel() == 0:
                intent_preds.append(None)
                motion_preds.append(None)
                continue

            # ROI Align expects [batch_index, x1, y1, x2, y2]
            rois = torch.cat([
                torch.full((boxes[b].size(0), 1), b, device=feat.device),
                boxes[b].to(feat.device)
            ], dim=1)

            roi_feat = ops.roi_align(
                feat,
                rois,
                output_size=self.roi_size,
                spatial_scale=1.0,
                aligned=True
            )

            roi_feat = roi_feat.view(roi_feat.size(0), -1)

            motion = self.motion_mlp(roi_feat)
            motion = motion.view(-1, self.future_frames, 2)

            intent = self.intent_mlp(roi_feat)

            motion_preds.append(motion)
            intent_preds.append(intent)

        return intent_preds, motion_preds