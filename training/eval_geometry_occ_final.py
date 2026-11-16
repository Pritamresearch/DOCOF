import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from data.occupancy_generator import OccupancyGenerator


# =====================================================
# CONFIG
# =====================================================

DATAROOT = r"D:\Pritam\TS BEV\NuScenes"
VERSION = "v1.0-trainval"
CHECKPOINT_PATH = "checkpoints/geometry_occ_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
SEQ_LEN = 3
THRESHOLD = 0.3

OUTPUT_DIR = "outputs/geometry"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        past = torch.stack(occ_seq[:self.seq_len])
        future = occ_seq[self.seq_len]

        return past.unsqueeze(1), future.unsqueeze(0)


# =====================================================
# MODEL
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

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

splits = create_splits_scenes()
val_scene_names = splits["val"]
val_scenes = [s for s in nusc.scene if s["name"] in val_scene_names]

val_dataset = OccupancySequenceDataset(nusc, val_scenes, SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =====================================================
# LOAD MODEL
# =====================================================

model = ConvGRU().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print("Stored Val IoU:", checkpoint["val_iou"])


# =====================================================
# METRICS
# =====================================================

ious = []
precisions = []
recalls = []
f1_scores = []
accuracies = []
gt_means = []
pred_means = []

with torch.no_grad():

    for past, future in tqdm(val_loader):

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        probs = torch.sigmoid(model(past))
        pred = (probs > THRESHOLD).float()

        TP = (pred * future).sum(dim=[1,2,3])
        FP = (pred * (1 - future)).sum(dim=[1,2,3])
        FN = ((1 - pred) * future).sum(dim=[1,2,3])
        TN = ((1 - pred) * (1 - future)).sum(dim=[1,2,3])

        intersection = TP
        union = TP + FP + FN

        iou = intersection / (union + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

        ious.extend(iou.cpu().numpy())
        precisions.extend(precision.cpu().numpy())
        recalls.extend(recall.cpu().numpy())
        f1_scores.extend(f1.cpu().numpy())
        accuracies.extend(accuracy.cpu().numpy())

        gt_means.append(future.mean().item())
        pred_means.append(probs.mean().item())


print("\n========== FINAL GEOMETRY RESULTS ==========")
print(f"Mean IoU        : {np.mean(ious):.4f}")
print(f"Precision       : {np.mean(precisions):.4f}")
print(f"Recall          : {np.mean(recalls):.4f}")
print(f"F1 Score        : {np.mean(f1_scores):.4f}")
print(f"Accuracy        : {np.mean(accuracies):.4f}")
print(f"GT Occupancy    : {np.mean(gt_means):.6f}")
print(f"Pred Occupancy  : {np.mean(pred_means):.6f}")
print("=============================================")


# =====================================================
# EXPORT VISUALS (FIXED LAYOUT)
# =====================================================

NUM_SAMPLES = 15  # how many images to export

with torch.no_grad():

    for i, (past, future) in enumerate(val_loader):

        if i >= NUM_SAMPLES:
            break

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        probs = torch.sigmoid(model(past))
        pred = (probs > THRESHOLD).float()

        past_np = past[0, -1, 0].cpu().numpy()
        future_np = future[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()

        overlay = np.zeros((future_np.shape[0], future_np.shape[1], 3))
        overlay[..., 0] = future_np   # Red = GT
        overlay[..., 1] = pred_np     # Green = Prediction

        # Bigger figure height
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(past_np, cmap='gray')
        axes[0].set_title("Last Past Occupancy", fontsize=12)

        axes[1].imshow(future_np, cmap='gray')
        axes[1].set_title("GT Future", fontsize=12)

        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title("Predicted Future", fontsize=12)

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (Red = GT, Green = Pred)", fontsize=12)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout(pad=2.0)

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"sample_{i}.png"),
            bbox_inches='tight',
            dpi=300
        )

        plt.close()

print("Visualization export complete.")
