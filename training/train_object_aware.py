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
LR = 1e-3
NUM_CLASSES = 3

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================
# DICE LOSS
# =====================================================

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
        loss = 1 - dice

        return loss.mean()


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
            grid = self.occ_gen.generate(sample)   # numpy [3,H,W]

            grid = torch.from_numpy(grid).float()

            # 🔥 FORCE SHAPE SAFETY
            assert grid.dim() == 3
            assert grid.shape[0] == 3

            occ_seq.append(grid)

            if sample["next"] == "":
                break

            token = sample["next"]

        while len(occ_seq) < self.seq_len + 1:
            occ_seq.append(torch.zeros_like(occ_seq[0]))

        past = torch.stack(occ_seq[:self.seq_len])   # [T,3,H,W]
        future = occ_seq[self.seq_len]               # [3,H,W]

        return past, future




# =====================================================
# MODEL
# =====================================================

class ConvGRU3Class(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.encoder = nn.Conv2d(NUM_CLASSES, channels, 3, padding=1)

        self.conv_z = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.conv_r = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.conv_h = nn.Conv2d(channels * 2, channels, 3, padding=1)

        self.decoder = nn.Conv2d(channels, NUM_CLASSES, 1)

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

            combined_r = torch.cat([r * h, xt], dim=1)
            h_tilde = torch.tanh(self.conv_h(combined_r))

            h = (1 - z) * h + z * h_tilde

        return self.decoder(h)


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
# TRAINING
# =====================================================

model = ConvGRU3Class().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

pos_weight = torch.tensor([1.0, 5.0, 3.0]).to(DEVICE)

class_weights = torch.tensor([1.0, 5.0, 3.0]).to(DEVICE)

bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
dice_loss = DiceLoss()



best_miou = 0.0

print("\nStarting Object-Aware 3-Class Training...\n")

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for past, future in tqdm(train_loader):

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        optimizer.zero_grad()

        logits = model(past)

        # Shape sanity check
        assert logits.shape == future.shape, \
            f"Shape mismatch: logits {logits.shape}, future {future.shape}"


        # BCE per pixel
        bce = bce_loss_fn(logits, future)  # [B,3,H,W]

        # Apply class weights manually
        for c in range(NUM_CLASSES):
            bce[:, c] *= class_weights[c]

        loss_bce = bce.mean()

        loss_dice = dice_loss(logits, future)

        loss = loss_bce + loss_dice


        
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

# Save final model
torch.save({
    "model_state_dict": model.state_dict(),
    "epoch": EPOCHS,
    "val_miou": best_miou
}, os.path.join(CHECKPOINT_DIR, "object_aware_final.pth"))

print("\nTraining complete.")
print(f"Best Val mIoU: {best_miou:.4f}")
