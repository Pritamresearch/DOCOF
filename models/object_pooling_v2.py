import torch
import torch.nn as nn


class ObjectRegionPooling(nn.Module):
    """
    Extract object tokens from BEV feature map
    using simple ROI average pooling.
    """

    def __init__(self):
        super().__init__()

    def forward(self, bev_feature_map, boxes):
        """
        bev_feature_map: [B, D, H, W]
        boxes: list of length B,
               each element is tensor [N, 4]
               format: (x_min, y_min, x_max, y_max) in grid coords

        Returns:
            object_tokens: list of [N_i, D]
        """

        B, D, H, W = bev_feature_map.shape
        all_tokens = []

        for b in range(B):

            feature = bev_feature_map[b]  # [D,H,W]
            obj_boxes = boxes[b]

            tokens = []

            for box in obj_boxes:

                x1, y1, x2, y2 = box.int()

                x1 = torch.clamp(x1, 0, W-1)
                x2 = torch.clamp(x2, 0, W-1)
                y1 = torch.clamp(y1, 0, H-1)
                y2 = torch.clamp(y2, 0, H-1)

                region = feature[:, y1:y2+1, x1:x2+1]

                if region.numel() == 0:
                    continue

                pooled = region.mean(dim=(1, 2))  # [D]
                tokens.append(pooled)

            if len(tokens) == 0:
                all_tokens.append(torch.zeros(1, D, device=bev_feature_map.device))
            else:
                all_tokens.append(torch.stack(tokens))

        return all_tokens