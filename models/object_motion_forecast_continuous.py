import torch
import torch.nn as nn


class ObjectMotionForecastContinuous(nn.Module):

    def __init__(self, future_frames=3, num_intent_classes=5):
        super().__init__()

        self.future_frames = future_frames
        self.num_intent_classes = num_intent_classes

        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.motion_head = nn.Linear(64, future_frames * 2)
        self.intent_head = nn.Linear(64, num_intent_classes)

    def forward(self, bev_past, boxes):

        # bev_past: [B, T, 1, H, W]
        B = bev_past.size(0)

        # 🔥 FIX: permute to [B, C, T, H, W]
        bev_past = bev_past.permute(0, 2, 1, 3, 4)

        features = self.temporal_encoder(bev_past).view(B, -1)

        intent_preds = []
        motion_preds = []

        for b in range(B):

            if boxes[b].numel() == 0:
                intent_preds.append(None)
                motion_preds.append(None)
                continue

            num_obj = boxes[b].size(0)

            feat = features[b].unsqueeze(0).repeat(num_obj, 1)

            motion = self.motion_head(feat)

            # Constrain displacement to [-10m, 10m]
            motion = torch.tanh(motion) * 10.0

            motion = motion.view(num_obj, self.future_frames, 2)

            intent = self.intent_head(feat)

            motion_preds.append(motion)
            intent_preds.append(intent)

        return intent_preds, motion_preds