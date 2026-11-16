import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from data.nuscenes_loader import NuScenesLoader
from data.occupancy_generator import OccupancyGenerator

from models.bev_encoder import BEVEncoder
from models.temporal_backbone import TemporalBackbone
from models.occupancy_head import OccupancyHead


# =========================================================
# DICE LOSS
# =========================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        return loss.mean()


# =========================================================
# DATASET
# =========================================================

class ForecastDataset(Dataset):
    def __init__(self, nusc, scenes, seq_len):
        self.nusc = nusc
        self.scenes = scenes
        self.seq_len = seq_len
        self.occ_gen = OccupancyGenerator(nusc)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        loader = NuScenesLoader(self.nusc, scene, self.seq_len)

    # print("Future sample token:", future_sample["token"])
    # print("Annotations:", len(future_sample["anns"]))




        for frames, future_sample in loader:
            frames_tensor = torch.stack(frames)   # [T,3,H,W]
            gt_grid = self.occ_gen.generate(future_sample)
            break

        gt_grid = torch.tensor(gt_grid).float()  # [H,W]
        return frames_tensor, gt_grid


# =========================================================
# MAIN
# =========================================================

def main():

    DATAROOT = r"D:\Pritam\TS BEV\NuScenes"
    VERSION = "v1.0-trainval"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    BATCH_SIZE = 8
    EPOCHS = 20
    SEQ_LEN = 3
    LR = 1e-4

    os.makedirs("checkpoints", exist_ok=True)

    # =====================================================
    # LOAD NUSCENES
    # =====================================================

    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    splits = create_splits_scenes()
    train_scene_names = splits["train"]
    val_scene_names = splits["val"]

    train_scenes = [s for s in nusc.scene if s["name"] in train_scene_names]
    val_scenes = [s for s in nusc.scene if s["name"] in val_scene_names]

    print(f"Train scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")

    # =====================================================
    # DATA
    # =====================================================

    train_dataset = ForecastDataset(nusc, train_scenes, SEQ_LEN)
    val_dataset = ForecastDataset(nusc, val_scenes, SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # =====================================================
    # MODELS
    # =====================================================

    bev_encoder = BEVEncoder().to(DEVICE)
    temporal = TemporalBackbone(channels=16).to(DEVICE)
    occ_head = OccupancyHead(in_channels=16).to(DEVICE)

    optimizer = Adam(
        list(bev_encoder.parameters()) +
        list(temporal.parameters()) +
        list(occ_head.parameters()),
        lr=LR
    )

    # IMPORTANT: reduce pos_weight
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(DEVICE))
    dice_loss = DiceLoss()

    scaler = GradScaler(device="cuda")
    best_val_iou = 0.0

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    print("\nStarting Geometry Training...\n")

    for epoch in range(EPOCHS):

        bev_encoder.train()
        temporal.train()
        occ_head.train()

        train_loss = 0.0

        for frames_batch, gt_batch in tqdm(train_loader):

            frames_batch = frames_batch.to(DEVICE)
            gt_batch = gt_batch.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):

                B, T, C, H, W = frames_batch.shape

                bev_seq = []
                for t in range(T):
                    bev_feat = bev_encoder(frames_batch[:, t])
                    bev_seq.append(bev_feat)

                bev_seq = torch.stack(bev_seq, dim=1)

                fused = temporal(bev_seq)
                logits = occ_head(fused)

                loss_bce = bce_loss(logits, gt_batch)
                loss_dice = dice_loss(logits, gt_batch)

                loss = loss_bce + loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # =====================================================
        # VALIDATION
        # =====================================================

        bev_encoder.eval()
        temporal.eval()
        occ_head.eval()

        val_ious = []

        with torch.no_grad():
            for frames_batch, gt_batch in val_loader:

                frames_batch = frames_batch.to(DEVICE)
                gt_batch = gt_batch.unsqueeze(1).to(DEVICE)

                B, T, C, H, W = frames_batch.shape

                bev_seq = []
                for t in range(T):
                    bev_feat = bev_encoder(frames_batch[:, t])
                    bev_seq.append(bev_feat)

                bev_seq = torch.stack(bev_seq, dim=1)

                fused = temporal(bev_seq)
                probs = torch.sigmoid(occ_head(fused))

                pred_bin = (probs > 0.01).float()

                intersection = (pred_bin * gt_batch).sum(dim=[1,2,3])
                union = (pred_bin + gt_batch - pred_bin * gt_batch).sum(dim=[1,2,3])

                iou = intersection / (union + 1e-6)
                val_ious.extend(iou.cpu().numpy())

                ##########################
            print("GT mean:", gt_batch.mean().item())
            print("Val GT occupancy:", gt_batch.sum().item())
            print("Pred mean:", probs.mean().item())

        mean_val_iou = np.mean(val_ious)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val IoU: {mean_val_iou:.4f}")

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            torch.save({
                "bev": bev_encoder.state_dict(),
                "temporal": temporal.state_dict(),
                "head": occ_head.state_dict(),
                "epoch": epoch,
                "val_iou": mean_val_iou
            }, "checkpoints/geometry_best.pth")

            print("Saved new best model.")

    print("\nTraining complete.")
    print(f"Best Val IoU: {best_val_iou:.4f}")


# =========================================================
# WINDOWS SAFE ENTRY
# =========================================================

if __name__ == "__main__":
    main()
