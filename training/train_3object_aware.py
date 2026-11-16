import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from data.object_occupancy_generator import ObjectOccupancyGenerator
from models.object_temporal_model import ObjectTemporalModel


# =====================================================
# CONFIG
# =====================================================

DATAROOT = r"D:\Pritam\TS BEV\NuScenes"
VERSION = "v1.0-trainval"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 8
EPOCHS = 40
SEQ_LEN = 3
LR = 3e-4
NUM_CLASSES = 3

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================
# FOCAL + DICE LOSS
# =====================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        B, C, H, W = probs.shape

        probs = probs.view(B, C, -1)
        targets = targets.view(B, C, -1)

        intersection = (probs * targets).sum(dim=2)
        union = probs.sum(dim=2) + targets.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# =====================================================
# DATASET
# =====================================================

class ObjectSequenceDataset(Dataset):

    def __init__(self, nusc, scenes, seq_len):
        self.nusc = nusc
        self.scenes = scenes
        self.seq_len = seq_len
        self.occ_gen = ObjectOccupancyGenerator(nusc)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):

        scene = self.scenes[idx]
        token = scene["first_sample_token"]

        occ_seq = []

        for _ in range(self.seq_len + 1):

            sample = self.nusc.get("sample", token)
            grid = self.occ_gen.generate(sample)

            grid = torch.from_numpy(grid).float()
            occ_seq.append(grid)

            if sample["next"] == "":
                break

            token = sample["next"]

        while len(occ_seq) < self.seq_len + 1:
            occ_seq.append(torch.zeros_like(occ_seq[0]))

        past = torch.stack(occ_seq[:self.seq_len])
        future = occ_seq[self.seq_len]

        return past, future


# =====================================================
# LOAD DATA
# =====================================================

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

splits = create_splits_scenes()
train_scenes = [s for s in nusc.scene if s["name"] in splits["train"]]
val_scenes = [s for s in nusc.scene if s["name"] in splits["val"]]

train_dataset = ObjectSequenceDataset(nusc, train_scenes, SEQ_LEN)
val_dataset = ObjectSequenceDataset(nusc, val_scenes, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =====================================================
# MODEL
# =====================================================

model = ObjectTemporalModel().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

focal_loss = FocalLoss()
dice_loss = DiceLoss()

best_miou = 0.0

print("\nStarting Strong Object-Aware Training...\n")


# =====================================================
# TRAIN LOOP
# =====================================================

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for past, future in tqdm(train_loader):

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        optimizer.zero_grad()

        logits = model(past)

        loss = focal_loss(logits, future) + dice_loss(logits, future)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ================= VALIDATION =================

    model.eval()
    class_ious = [[] for _ in range(NUM_CLASSES)]

    with torch.no_grad():

        for past, future in val_loader:

            past = past.to(DEVICE)
            future = future.to(DEVICE)

            logits = model(past)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.3).float()

            for c in range(NUM_CLASSES):

                p = pred[:, c]
                gt = future[:, c]

                intersection = (p * gt).sum(dim=[1,2])
                union = (p + gt - p * gt).sum(dim=[1,2])

                iou = intersection / (union + 1e-6)
                class_ious[c].extend(iou.cpu().numpy())

    miou = np.mean([np.mean(class_ious[c]) for c in range(NUM_CLASSES)])

    print(f"\nEpoch {epoch}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val mIoU : {miou:.4f}")

    if miou > best_miou:
        best_miou = miou
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_miou": best_miou
        }, os.path.join(CHECKPOINT_DIR, "object_aware_best.pth"))
        print("Saved NEW BEST model.")

print("\nTraining complete.")
print(f"Best Val mIoU: {best_miou:.4f}")
