import torch
import torch.nn as nn
import torchvision.ops as ops


class ObjectMotionForecast(nn.Module):

    def __init__(self, d_model=64, future_frames=3, num_intent_classes=3):
        super().__init__()

        self.future_frames = future_frames

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, d_model, 3, padding=1),
            nn.ReLU()
        )

        self.roi_align = ops.RoIAlign(
            output_size=(3, 3),
            spatial_scale=1.0,
            sampling_ratio=-1
        )

        self.obj_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 3 * 3, 128),
            nn.ReLU()
        )

        self.intent_head = nn.Linear(128, num_intent_classes)
        self.motion_head = nn.Linear(128, 2 * future_frames)

    def forward(self, bev_past, boxes):

        B, T, C, H, W = bev_past.shape

        x = bev_past.permute(0, 2, 1, 3, 4)
        feat = self.encoder(x)
        feat = feat.mean(dim=2)

        all_intents = []
        all_motion = []

        for b in range(B):

            if boxes[b].numel() == 0:
                all_intents.append(None)
                all_motion.append(None)
                continue

            num_boxes = boxes[b].shape[0]

            box_tensor = boxes[b].to(feat.device)   # 🔥 MOVE TO GPU

            rois = torch.cat([
                torch.full((num_boxes, 1), b, device=feat.device),
                box_tensor
            ], dim=1)

            obj_feat = self.roi_align(feat, rois)
            obj_feat = self.obj_fc(obj_feat)

            intent_logits = self.intent_head(obj_feat)
            motion = self.motion_head(obj_feat)
            motion = motion.view(num_boxes, self.future_frames, 2)

            all_intents.append(intent_logits)
            all_motion.append(motion)

        return all_intents, all_motion