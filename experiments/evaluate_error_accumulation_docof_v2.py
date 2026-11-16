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
OUTPUT_DIR = "error_accumulation_results"
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
# Accumulators
# ---------------------------------------

ade_sum = torch.zeros(FUTURE_FRAMES)
fde_sum = 0.0
total_valid = torch.zeros(FUTURE_FRAMES)


# ---------------------------------------
# Evaluation Loop
# ---------------------------------------

with torch.no_grad():

    for idx in tqdm(range(len(val_dataset)), desc="ADE Eval"):

        sample = val_dataset[idx]

        history = sample["history"].unsqueeze(0).to(DEVICE)
        state_t = sample["state_t"].unsqueeze(0).to(DEVICE)
        masks = [
            m.unsqueeze(0).to(DEVICE)
            for m in sample["validity_masks"]
        ]

        future_gt = [
            f.unsqueeze(0).to(DEVICE)
            for f in sample["future_states"]
        ]

        states_pred, _, _ = model(
            history,
            state_t,
            masks
        )

        for k in range(FUTURE_FRAMES):

            # Position error
            error = torch.norm(
                states_pred[k][:, :, :2] - future_gt[k][:, :, :2],
                dim=-1
            )

            valid_mask = masks[k]

            ade_sum[k] += (error * valid_mask).sum().cpu()
            total_valid[k] += valid_mask.sum().cpu()

            # FDE at final horizon
            if k == FUTURE_FRAMES - 1:
                fde_sum += (error * valid_mask).sum().cpu()


# ---------------------------------------
# Normalize
# ---------------------------------------

ade_per_horizon = ade_sum / (total_valid + 1e-6)
fde_final = fde_sum / (total_valid[-1] + 1e-6)


# ---------------------------------------
# Linear Fit
# ---------------------------------------

horizons = np.arange(1, FUTURE_FRAMES + 1)
ade_numpy = ade_per_horizon.numpy()

coeff = np.polyfit(horizons, ade_numpy, 1)
slope = coeff[0]


# ---------------------------------------
# Print Results
# ---------------------------------------

print("\n===== Error Accumulation Analysis (DOCOF v2) =====")

for k in range(FUTURE_FRAMES):
    print(f"Horizon +{k+1}: ADE = {ade_per_horizon[k]:.4f}")

print(f"\nFDE@{FUTURE_FRAMES}: {fde_final:.4f}")
print(f"Linear ADE Growth Slope: {slope:.4f}")


# ---------------------------------------
# Plot ADE Growth
# ---------------------------------------

plt.figure(figsize=(6,4))
plt.plot(horizons, ade_numpy, marker='o')
plt.xlabel("Future Horizon")
plt.ylabel("Average Displacement Error (m)")
plt.title("ADE Growth Across Horizons")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ade_growth.png"))
plt.close()


# ---------------------------------------
# Plot Error Curve
# ---------------------------------------

plt.figure(figsize=(6,4))
plt.plot(horizons, ade_numpy, marker='o')
plt.xlabel("Future Horizon")
plt.ylabel("Displacement Error (m)")
plt.title("Error Growth Behavior")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_growth.png"))
plt.close()

print(f"\nPlots saved in {OUTPUT_DIR}/")