import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from data.loader import NuScenesIntentDataset
from models.object_motion_forecast_v2 import ObjectMotionForecastV2
from utils.render_future import render_future_from_motion


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATAROOT = "D:/Pritam/TS BEV/NuScenes"
CHECKPOINT_DIR = "checkpoints"

FUTURE_FRAMES = 4
PAST_FRAMES = 5
GRID_SIZE = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs/stability_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

nusc = NuScenes("v1.0-trainval", DATAROOT, verbose=False)
splits = create_splits_scenes()

def collect_tokens(scene_names):
    tokens = []
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue
        token = scene["first_sample_token"]
        while token != "":
            tokens.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"]
    return tokens

val_tokens = collect_tokens(splits["val"])

dataset = NuScenesIntentDataset(
    nusc,
    val_tokens,
    past_frames=PAST_FRAMES,
    future_frames=FUTURE_FRAMES
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda b: {
        "bev_past": torch.stack([x["bev_past"] for x in b]),
        "bev_future": torch.stack([x["bev_future"] for x in b]),
        "boxes": [x["boxes"] for x in b],
        "motion_target": [x["motion_target"] for x in b]
    }
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = ObjectMotionForecastV2(
    future_frames=FUTURE_FRAMES,
    num_intent_classes=5
).to(device)

model.load_state_dict(
    torch.load(
        os.path.join(
            CHECKPOINT_DIR,
            f"object_motion_v2_K{FUTURE_FRAMES}_best.pth"
        ),
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Metrics Containers
# --------------------------------------------------

iou_per_horizon = np.zeros(FUTURE_FRAMES)
ade_per_horizon = np.zeros(FUTURE_FRAMES)
fde_per_horizon = np.zeros(FUTURE_FRAMES)
mass_pred = np.zeros(FUTURE_FRAMES)
mass_gt = np.zeros(FUTURE_FRAMES)
motion_var = np.zeros(FUTURE_FRAMES)

batches = 0
total_objects = 0


# --------------------------------------------------
# Evaluation Loop
# --------------------------------------------------

with torch.no_grad():

    for batch in loader:

        bev_past = batch["bev_past"].to(device)
        bev_future = batch["bev_future"].to(device)
        boxes = batch["boxes"]
        motion_target = batch["motion_target"]

        intent_pred, motion_pred = model(bev_past, boxes)

        rendered = render_future_from_motion(
            boxes,
            motion_pred,
            grid_size=GRID_SIZE,
            future_frames=FUTURE_FRAMES
        ).to(device)

        pred = torch.sigmoid(rendered)
        pred_bin = (pred > 0.5).float()

        # ----- IoU -----
        for k in range(FUTURE_FRAMES):

            p = pred_bin[:, k]
            t = bev_future[:, k]

            TP = (p * t).sum()
            FP = (p * (1 - t)).sum()
            FN = ((1 - p) * t).sum()

            iou = TP / (TP + FP + FN + 1e-6)
            iou_per_horizon[k] += iou.item()

            mass_pred[k] += p.sum().item()
            mass_gt[k] += t.sum().item()

        # ----- Motion errors -----
        for b in range(len(motion_pred)):

            if motion_pred[b] is None:
                continue

            pred_m = motion_pred[b].cpu().numpy()
            gt_m = motion_target[b].cpu().numpy()

            num_obj = pred_m.shape[0]
            total_objects += num_obj

            for o in range(num_obj):

                for k in range(FUTURE_FRAMES):
                    diff = pred_m[o, k] - gt_m[o, k]
                    dist = np.linalg.norm(diff)

                    ade_per_horizon[k] += dist
                    motion_var[k] += dist ** 2

                fde_per_horizon[FUTURE_FRAMES - 1] += \
                    np.linalg.norm(pred_m[o, FUTURE_FRAMES - 1] -
                                   gt_m[o, FUTURE_FRAMES - 1])

        batches += 1


# --------------------------------------------------
# Normalize
# --------------------------------------------------

iou_per_horizon /= batches
mass_pred /= batches
mass_gt /= batches

ade_per_horizon /= total_objects
fde_per_horizon /= total_objects
motion_var = np.sqrt(motion_var / total_objects)


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\n===== Long-Horizon Stability Analysis =====")
for k in range(FUTURE_FRAMES):
    print(f"Horizon +{k+1}")
    print(f" IoU: {iou_per_horizon[k]:.4f}")
    print(f" ADE: {ade_per_horizon[k]:.4f}")
    print(f" Motion StdDev: {motion_var[k]:.4f}")
    print()

print(f"FDE@{FUTURE_FRAMES}: {fde_per_horizon[FUTURE_FRAMES-1]:.4f}")


# --------------------------------------------------
# Plot Curves
# --------------------------------------------------

horizons = np.arange(1, FUTURE_FRAMES + 1)

# IoU degradation
plt.figure()
plt.plot(horizons, iou_per_horizon)
plt.xlabel("Horizon")
plt.ylabel("IoU")
plt.title("IoU Degradation Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "iou_curve.png"))
plt.close()

# ADE growth
plt.figure()
plt.plot(horizons, ade_per_horizon)
plt.xlabel("Horizon")
plt.ylabel("ADE")
plt.title("ADE Growth Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "ade_curve.png"))
plt.close()

# Motion variance
plt.figure()
plt.plot(horizons, motion_var)
plt.xlabel("Horizon")
plt.ylabel("Motion StdDev")
plt.title("Motion Uncertainty Growth")
plt.savefig(os.path.join(OUTPUT_DIR, "motion_variance.png"))
plt.close()

# Mass conservation
plt.figure()
plt.plot(horizons, mass_pred, label="Predicted")
plt.plot(horizons, mass_gt, label="Ground Truth")
plt.xlabel("Horizon")
plt.ylabel("Occupancy Mass")
plt.title("Mass Conservation Across Horizons")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "mass_conservation.png"))
plt.close()

print("\nAll stability plots saved in outputs/stability_full/")