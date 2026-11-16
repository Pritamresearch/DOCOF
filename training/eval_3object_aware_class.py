import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from data.object_occupancy_generator import ObjectOccupancyGenerator

# IMPORTANT: import model from models folder
from models.object_temporal_model import ObjectTemporalModel


# =====================================================
# CONFIG
# =====================================================

DATAROOT = r"D:\Pritam\TS BEV\NuScenes"
VERSION = "v1.0-trainval"
CHECKPOINT_PATH = "checkpoints/object_aware_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
SEQ_LEN = 3
THRESHOLD = 0.3
NUM_CLASSES = 3

OUTPUT_DIR = "outputs/object_aware_3class"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
            grid = self.occ_gen.generate(sample)  # numpy [3,H,W]

            grid = torch.from_numpy(grid).float()
            occ_seq.append(grid)

            if sample["next"] == "":
                break

            token = sample["next"]

        while len(occ_seq) < self.seq_len + 1:
            occ_seq.append(torch.zeros_like(occ_seq[0]))

        past = torch.stack(occ_seq[:self.seq_len])  # [T,3,H,W]
        future = occ_seq[self.seq_len]              # [3,H,W]

        return past, future


# =====================================================
# LOAD DATA
# =====================================================

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

splits = create_splits_scenes()
val_scenes = [s for s in nusc.scene if s["name"] in splits["val"]]

val_dataset = ObjectSequenceDataset(nusc, val_scenes, SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =====================================================
# LOAD MODEL
# =====================================================

model = ObjectTemporalModel().to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print("Stored validation mIoU:", checkpoint.get("val_miou", "Not stored"))


# =====================================================
# METRICS
# =====================================================

all_ious = [[] for _ in range(NUM_CLASSES)]
all_precisions = [[] for _ in range(NUM_CLASSES)]
all_recalls = [[] for _ in range(NUM_CLASSES)]
all_f1 = [[] for _ in range(NUM_CLASSES)]
all_accuracy = [[] for _ in range(NUM_CLASSES)]

gt_density = [[] for _ in range(NUM_CLASSES)]
pred_density = [[] for _ in range(NUM_CLASSES)]

with torch.no_grad():

    for past, future in tqdm(val_loader):

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        logits = model(past)
        probs = torch.sigmoid(logits)
        pred = (probs > THRESHOLD).float()

        for c in range(NUM_CLASSES):

            p = pred[:, c]
            gt = future[:, c]

            TP = (p * gt).sum(dim=[1,2])
            FP = (p * (1 - gt)).sum(dim=[1,2])
            FN = ((1 - p) * gt).sum(dim=[1,2])
            TN = ((1 - p) * (1 - gt)).sum(dim=[1,2])

            intersection = TP
            union = TP + FP + FN

            iou = intersection / (union + 1e-6)
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)

            all_ious[c].extend(iou.cpu().numpy())
            all_precisions[c].extend(precision.cpu().numpy())
            all_recalls[c].extend(recall.cpu().numpy())
            all_f1[c].extend(f1.cpu().numpy())
            all_accuracy[c].extend(acc.cpu().numpy())

            gt_density[c].append(gt.mean().item())
            pred_density[c].append(probs[:, c].mean().item())


# =====================================================
# PRINT RESULTS
# =====================================================

class_names = ["Vehicle", "Pedestrian", "Other"]

print("\n========== 3-CLASS OBJECT-AWARE RESULTS ==========")

for c in range(NUM_CLASSES):
    print(f"\nClass: {class_names[c]}")
    print(f"IoU        : {np.mean(all_ious[c]):.4f}")
    print(f"Precision  : {np.mean(all_precisions[c]):.4f}")
    print(f"Recall     : {np.mean(all_recalls[c]):.4f}")
    print(f"F1 Score   : {np.mean(all_f1[c]):.4f}")
    print(f"Accuracy   : {np.mean(all_accuracy[c]):.4f}")
    print(f"GT Density : {np.mean(gt_density[c]):.6f}")
    print(f"Pred Density: {np.mean(pred_density[c]):.6f}")

mean_miou = np.mean([np.mean(all_ious[c]) for c in range(NUM_CLASSES)])

print("\nOverall mIoU:", round(mean_miou, 4))
print("===================================================")


# =====================================================
# VISUALIZATION EXPORT
# =====================================================

NUM_SAMPLES = 10

with torch.no_grad():

    for i, (past, future) in enumerate(val_loader):

        if i >= NUM_SAMPLES:
            break

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        probs = torch.sigmoid(model(past))
        pred = (probs > THRESHOLD).float()

        fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(15, 6))

        for c in range(NUM_CLASSES):

            axes[0, c].imshow(future[0, c].cpu(), cmap='gray')
            axes[0, c].set_title(f"GT - {class_names[c]}")
            axes[0, c].axis("off")

            axes[1, c].imshow(pred[0, c].cpu(), cmap='gray')
            axes[1, c].set_title(f"Pred - {class_names[c]}")
            axes[1, c].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"sample_{i}.png"), dpi=300)
        plt.close()

print("Visualization export complete.")
