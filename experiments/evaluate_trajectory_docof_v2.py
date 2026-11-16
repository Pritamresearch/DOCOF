import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config_v2 import CONFIG_V2
from models.docof_model_v2 import DOCOFv2
from data.docof_torch_dataset_v2 import DOCOFTorchDatasetV2


# ---------------------------------------
# CONFIG
# ---------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "docof_v2_best_mean.pth"
OUTPUT_DIR = "trajectory_stability_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FUTURE_FRAMES = CONFIG_V2["future_frames"]
NUM_OBJECTS_TO_PLOT = 5


# ---------------------------------------
# Dataset
# ---------------------------------------

val_dataset = DOCOFTorchDatasetV2(
    root="preprocessed_docof_v2",
    split="val"
)

print(f"Loaded {len(val_dataset)} validation samples")


# ---------------------------------------
# Model
# ---------------------------------------

model = DOCOFv2(CONFIG_V2).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()


# ---------------------------------------
# Visualization
# ---------------------------------------

with torch.no_grad():

    for idx in range(len(val_dataset)):

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

        # Convert to numpy
        valid_mask = masks[0].squeeze(0).cpu().numpy()

        valid_indices = np.where(valid_mask > 0.5)[0]

        if len(valid_indices) < NUM_OBJECTS_TO_PLOT:
            continue

        selected_indices = valid_indices[:NUM_OBJECTS_TO_PLOT]

        plt.figure(figsize=(6, 6))

        for obj_idx in selected_indices:

            # Collect GT trajectory
            gt_traj = []
            for k in range(FUTURE_FRAMES):
                gt_traj.append(
                    future_gt[k][0, obj_idx, :2].cpu().numpy()
                )
            gt_traj = np.stack(gt_traj)

            # Collect predicted trajectory
            pred_traj = []
            for k in range(FUTURE_FRAMES):
                pred_traj.append(
                    states_pred[k][0, obj_idx, :2].cpu().numpy()
                )
            pred_traj = np.stack(pred_traj)

            # Plot GT
            plt.plot(
                gt_traj[:, 0],
                gt_traj[:, 1],
                marker="o",
                linewidth=2
            )

            # Plot Prediction
            plt.plot(
                pred_traj[:, 0],
                pred_traj[:, 1],
                marker="x",
                linestyle="--",
                linewidth=2
            )

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Trajectory Drift Visualization (DOCOF v2)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend(["GT", "Pred"], loc="best")

        save_path = os.path.join(
            OUTPUT_DIR,
            "trajectory_drift_docof_v2.png"
        )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Trajectory plot saved to {save_path}")
        break