import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes

from data.occupancy_generator import OccupancyGenerator
from models.bev_encoder import BEVEncoder
from models.temporal_backbone import TemporalBackbone
from models.occupancy_head import OccupancyHead


# =====================
# CONFIG
# =====================
DATAROOT = "dataset/nuScenes mini"
VERSION = "v1.0-mini"

SEQ_LEN = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5


# =====================
# LOAD DATA
# =====================
nusc = NuScenes(
    version=VERSION,
    dataroot=DATAROOT,
    verbose=True
)

occ_gen = OccupancyGenerator(grid_size=128)


# =====================
# LOAD MODELS
# =====================
bev_encoder = BEVEncoder().to(DEVICE)
temporal = TemporalBackbone().to(DEVICE)
occ_head = OccupancyHead(in_channels=16).to(DEVICE)

ckpt = torch.load("checkpoints/forecasting_mini_25epochs.pth", map_location=DEVICE)

bev_encoder.load_state_dict(ckpt["bev"])
temporal.load_state_dict(ckpt["temporal"])
occ_head.load_state_dict(ckpt["head"])

print("Checkpoint loaded.")


bev_encoder.eval()
temporal.eval()
occ_head.eval()


# =====================
# IoU FUNCTION
# =====================
def compute_iou(pred, gt):
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()

    if union == 0:
        return 0.0

    return intersection / union


# =====================
# EVALUATION LOOP
# =====================
ious = []

with torch.no_grad():

    for i in range(10):

        sample = nusc.sample[i]
        sample_token = sample["token"]

        # -------- collect past frames --------
        frames = []
        current = sample

        for _ in range(SEQ_LEN):
            sd_token = current["data"]["CAM_FRONT"]
            sd = nusc.get("sample_data", sd_token)

            img_path = f"{DATAROOT}/{sd['filename']}"
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img)

            if current["prev"] == "":
                break
            current = nusc.get("sample", current["prev"])

        frames = frames[::-1]

        bev_seq = []

        for img in frames:
            img = img.unsqueeze(0).to(DEVICE)
            bev_feat = bev_encoder(img)
            bev_seq.append(bev_feat)

        ts_bev = temporal(bev_seq)

        pred_logits = occ_head(ts_bev)
        pred_prob = torch.sigmoid(pred_logits)
        pred_binary = (pred_prob > THRESH).float()

        # -------- GT --------
        gt = occ_gen.generate_future_occupancy(
            nusc,
            sample_token,
            horizon=1
        )

        gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # -------- IoU --------
        pred_np = pred_binary.cpu().numpy().astype(bool)
        gt_np = gt.cpu().numpy().astype(bool)

        iou = compute_iou(pred_np, gt_np)
        ious.append(iou)

        # -------- visualize first example --------
        if i == 0:
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.title("Ground Truth")
            plt.imshow(gt_np[0, 0], cmap="gray")

            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            plt.imshow(pred_np[0, 0], cmap="gray")

            plt.show()

mean_iou = np.mean(ious)
print(f"\nMean IoU: {mean_iou:.4f}")
