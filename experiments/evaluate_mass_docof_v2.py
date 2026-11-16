import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config_v2 import CONFIG_V2
from models.docof_model_v2 import DOCOFv2
from data.docof_torch_dataset_v2 import DOCOFTorchDatasetV2


# ---------------------------------------
# Configuration
# ---------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "docof_v2_best_mean.pth"
OUTPUT_DIR = "mass_conservation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FUTURE_FRAMES = CONFIG_V2["future_frames"]


# ---------------------------------------
# Dataset
# ---------------------------------------

val_dataset = DOCOFTorchDatasetV2(
    root="preprocessed_docof_v2",
    split="val"
)

print(f"Evaluating mass conservation on {len(val_dataset)} samples")


# ---------------------------------------
# Model
# ---------------------------------------

model = DOCOFv2(CONFIG_V2).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()


# ---------------------------------------
# Mass Accumulators
# ---------------------------------------

pred_mass = torch.zeros(FUTURE_FRAMES)
gt_mass = torch.zeros(FUTURE_FRAMES)


# ---------------------------------------
# Evaluation Loop
# ---------------------------------------

with torch.no_grad():

    for idx in tqdm(range(len(val_dataset)), desc="Mass Eval"):

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

        states_pred, occ_pred, _ = model(
            history,
            state_t,
            masks
        )

        for k in range(FUTURE_FRAMES):

            # Apply same threshold as trainer
            pred_bin = (occ_pred[k].squeeze(0) > 0.3).float()

            pred_mass[k] += pred_bin.sum().cpu()
            gt_mass[k] += occ_gt[k].sum().cpu()


# ---------------------------------------
# Normalize
# ---------------------------------------

pred_mass /= len(val_dataset)
gt_mass /= len(val_dataset)

relative_error = torch.abs(pred_mass - gt_mass) / (gt_mass + 1e-6)


# ---------------------------------------
# Print Summary
# ---------------------------------------

print("\n===== Occupancy Mass Conservation (DOCOF v2) =====")

for k in range(FUTURE_FRAMES):
    print(
        f"Horizon +{k+1}: "
        f"Pred={pred_mass[k]:.2f} | "
        f"GT={gt_mass[k]:.2f} | "
        f"RelErr={relative_error[k]*100:.2f}%"
    )


# ---------------------------------------
# Plot
# ---------------------------------------

horizons = np.arange(1, FUTURE_FRAMES + 1)

plt.figure(figsize=(6,4))
plt.plot(horizons, pred_mass.numpy(), marker='o')
plt.plot(horizons, gt_mass.numpy(), marker='o')
plt.xlabel("Future Horizon")
plt.ylabel("Occupancy Mass")
plt.title("Occupancy Mass Conservation (DOCOF v2)")
plt.legend(["Predicted", "Ground Truth"])
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mass_conservation_curve.png"))
plt.close()

print(f"\nPlot saved to {OUTPUT_DIR}/mass_conservation_curve.png")