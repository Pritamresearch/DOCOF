import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config_v2 import CONFIG_V2
from models.docof_model_v2 import DOCOFv2
from data.docof_torch_dataset_v2 import DOCOFTorchDatasetV2


# ---------------------------------------
# CONFIG
# ---------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "docof_v2_best_mean.pth"
OUTPUT_DIR = "horizon_degradation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FUTURE_FRAMES = CONFIG_V2["future_frames"]


# ---------------------------------------
# Dataset
# ---------------------------------------

val_dataset = DOCOFTorchDatasetV2(
    root="preprocessed_docof_v2",
    split="val"
)

print(f"Evaluating {len(val_dataset)} validation samples")


# ---------------------------------------
# Model
# ---------------------------------------

model = DOCOFv2(CONFIG_V2).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()


# ---------------------------------------
# IoU Accumulators (Global)
# ---------------------------------------

total_intersection = torch.zeros(FUTURE_FRAMES)
total_union = torch.zeros(FUTURE_FRAMES)


# ---------------------------------------
# Evaluation Loop
# ---------------------------------------

with torch.no_grad():

    for idx in tqdm(range(len(val_dataset)), desc="IoU Eval"):

        sample = val_dataset[idx]

        history = sample["history"].unsqueeze(0).to(DEVICE)
        state_t = sample["state_t"].unsqueeze(0).to(DEVICE)
        masks = [
            m.unsqueeze(0).to(DEVICE)
            for m in sample["validity_masks"]
        ]

        occ_gt = [
            o.to(DEVICE)
            for o in sample["future_occupancy"]
        ]

        _, occ_pred, _ = model(history, state_t, masks)

        for k in range(FUTURE_FRAMES):

            pred_bin = (occ_pred[k].squeeze(0) > 0.3).float()
            gt_bin = (occ_gt[k] > 0.5).float()

            intersection = (pred_bin * gt_bin).sum()
            union = ((pred_bin + gt_bin) > 0).float().sum()

            total_intersection[k] += intersection.cpu()
            total_union[k] += union.cpu()


# ---------------------------------------
# Compute IoU per Horizon
# ---------------------------------------

iou_per_horizon = total_intersection / (total_union + 1e-6)
iou_np = iou_per_horizon.numpy()


# ---------------------------------------
# Compute Δ_k
# ---------------------------------------

delta = iou_np[:-1] - iou_np[1:]
mean_delta = delta.mean()
std_delta = delta.std()


# ---------------------------------------
# Print Results
# ---------------------------------------

print("\n===== Horizon Degradation Analysis (DOCOF v2) =====")

for k in range(FUTURE_FRAMES):
    print(f"Horizon +{k+1}: IoU = {iou_np[k]:.4f}")

print("\nΔ_k (IoU_k - IoU_k+1):")
for k in range(len(delta)):
    print(f"Δ_{k+1} = {delta[k]:.4f}")

print(f"\nMean Δ: {mean_delta:.4f}")
print(f"Std  Δ: {std_delta:.4f}")


# ---------------------------------------
# Plot Curve
# ---------------------------------------

horizons = np.arange(1, FUTURE_FRAMES + 1)

plt.figure(figsize=(6,4))
plt.plot(horizons, iou_np, marker='o')
plt.xlabel("Future Horizon")
plt.ylabel("IoU")
plt.title("Horizon Degradation Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "iou_degradation_curve.png"))
plt.close()

print(f"\nPlot saved to {OUTPUT_DIR}/iou_degradation_curve.png")