import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os

from nuscenes.nuscenes import NuScenes

from data.occupancy_generator import OccupancyGenerator
from models.bev_encoder import BEVEncoder
from models.temporal_backbone import TemporalBackbone
from models.occupancy_head import OccupancyHead


# =====================
# CONFIG
# =====================
VERSION = "v1.0-trainval"
DATAROOT = r"D:\Pritam\TS BEV\NuScenes"


SEQ_LEN = 3
EPOCHS = 25
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
# MODELS
# =====================
bev_encoder = BEVEncoder().to(DEVICE)
temporal = TemporalBackbone().to(DEVICE)
occ_head = OccupancyHead(in_channels=16).to(DEVICE)

optimizer = optim.Adam(
    list(bev_encoder.parameters()) +
    list(temporal.parameters()) +
    list(occ_head.parameters()),
    lr=LR
)

pos_weight = torch.tensor([20.0]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



# =====================
# TRAIN LOOP
# =====================
print("Starting forecasting training...")

for epoch in range(EPOCHS):

    total_loss = 0.0

    for i in range(10):  # small loop for mini debug

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

        frames = frames[::-1]  # chronological order

        bev_seq = []

        for img in frames:
            img = img.unsqueeze(0).to(DEVICE)
            bev_feat = bev_encoder(img)
            bev_seq.append(bev_feat)

        ts_bev = temporal(bev_seq)

        # -------- ground truth occupancy --------
        gt = occ_gen.generate_future_occupancy(
            nusc,
            sample_token,
            horizon=1
        )

        gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # -------- prediction --------
        pred = occ_head(ts_bev)

        loss = criterion(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss/10:.4f}")


os.makedirs("checkpoints", exist_ok=True)

torch.save({
    "bev": bev_encoder.state_dict(),
    "temporal": temporal.state_dict(),
    "head": occ_head.state_dict()
}, "checkpoints/forecasting_mini_25epochs.pth")

print("Checkpoint saved.")


