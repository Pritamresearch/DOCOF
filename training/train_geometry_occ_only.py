import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from data.occupancy_generator import OccupancyGenerator


# =====================================================
# CONFIG
# =====================================================

DATAROOT = r"D:\Pritam\TS BEV\NuScenes"
VERSION = "v1.0-trainval"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 8
EPOCHS = 30
SEQ_LEN = 3
LR = 1e-3

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================
# DATASET
# =====================================================

class OccupancySequenceDataset(Dataset):

    def __init__(self, nusc, scenes, seq_len):
        self.nusc = nusc
        self.scenes = scenes
        self.seq_len = seq_len
        self.occ_gen = OccupancyGenerator(nusc)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):

        scene = self.scenes[idx]
        sample_token = scene["first_sample_token"]

        occ_seq = []

        for _ in range(self.seq_len + 1):

            sample = self.nusc.get("sample", sample_token)
            grid = self.occ_gen.generate(sample)
            occ_seq.append(torch.tensor(grid).float())

            if sample["next"] == "":
                break
            sample_token = sample["next"]

        if len(occ_seq) < self.seq_len + 1:
            while len(occ_seq) < self.seq_len + 1:
                occ_seq.append(torch.zeros_like(occ_seq[0]))

        past = torch.stack(occ_seq[:self.seq_len])   # [T,H,W]
        future = occ_seq[self.seq_len]               # [H,W]

        return past.unsqueeze(1), future.unsqueeze(0)


# =====================================================
# TEMPORAL MODEL
# =====================================================

class ConvGRU(nn.Module):

    def __init__(self, channels=16):
        super().__init__()

        self.encoder = nn.Conv2d(1, channels, 3, padding=1)

        self.conv_z = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv_r = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv_h = nn.Conv2d(channels*2, channels, 3, padding=1)

        self.decoder = nn.Conv2d(channels, 1, 1)

    def forward(self, x):

        B, T, C, H, W = x.shape
        h = None

        for t in range(T):

            xt = self.encoder(x[:, t])

            if h is None:
                h = xt
                continue

            combined = torch.cat([h, xt], dim=1)

            z = torch.sigmoid(self.conv_z(combined))
            r = torch.sigmoid(self.conv_r(combined))

            combined_r = torch.cat([r*h, xt], dim=1)
            h_tilde = torch.tanh(self.conv_h(combined_r))

            h = (1 - z) * h + z * h_tilde

        return self.decoder(h)


# =====================================================
# LOAD DATA
# =====================================================

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

splits = create_splits_scenes()
train_scene_names = splits["train"]
val_scene_names = splits["val"]

train_scenes = [s for s in nusc.scene if s["name"] in train_scene_names]
val_scenes = [s for s in nusc.scene if s["name"] in val_scene_names]

train_dataset = OccupancySequenceDataset(nusc, train_scenes, SEQ_LEN)
val_dataset = OccupancySequenceDataset(nusc, val_scenes, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =====================================================
# MODEL
# =====================================================

model = ConvGRU().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

best_iou = 0.0


# =====================================================
# TRAIN LOOP
# =====================================================

print("\nStarting Geometry Occupancy Training...\n")

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for past, future in tqdm(train_loader):

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        optimizer.zero_grad()

        logits = model(past)
        loss = criterion(logits, future)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ================= Validation =================

    model.eval()
    ious = []

    with torch.no_grad():
        for past, future in val_loader:

            past = past.to(DEVICE)
            future = future.to(DEVICE)

            probs = torch.sigmoid(model(past))
            pred = (probs > 0.3).float()

            intersection = (pred * future).sum(dim=[1,2,3])
            union = (pred + future - pred*future).sum(dim=[1,2,3])

            iou = intersection / (union + 1e-6)
            ious.extend(iou.cpu().numpy())

    mean_iou = np.mean(ious)

    print(f"\nEpoch {epoch}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val IoU: {mean_iou:.4f}")

    # Save best
    if mean_iou > best_iou:
        best_iou = mean_iou

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_iou": best_iou
        }, os.path.join(CHECKPOINT_DIR, "geometry_occ_best.pth"))

        print("Saved new BEST model.")

# Save final model
torch.save({
    "model_state_dict": model.state_dict(),
    "epoch": EPOCHS,
    "val_iou": best_iou
}, os.path.join(CHECKPOINT_DIR, "geometry_occ_final.pth"))

print("\nTraining complete.")
print(f"Best Val IoU: {best_iou:.4f}")
