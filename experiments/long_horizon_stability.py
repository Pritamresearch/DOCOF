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
GRID_SIZE = 200
PAST_FRAMES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs/stability"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load nuScenes Validation Split
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
# Metrics
# --------------------------------------------------

def compute_iou_per_horizon(pred, target):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    K = pred.shape[1]
    ious = []

    for k in range(K):

        p = pred[:, k]
        t = target[:, k]

        TP = (p * t).sum()
        FP = (p * (1 - t)).sum()
        FN = ((1 - p) * t).sum()

        iou = TP / (TP + FP + FN + 1e-6)
        ious.append(iou.item())

    return ious


def compute_motion_errors(motion_pred, motion_target):

    K = FUTURE_FRAMES

    ade = np.zeros(K)
    fde = np.zeros(K)
    count = 0

    for b in range(len(motion_pred)):

        if motion_pred[b] is None:
            continue

        pred = motion_pred[b].cpu().numpy()
        gt = motion_target[b].cpu().numpy()

        num_obj = pred.shape[0]

        for o in range(num_obj):

            for k in range(K):
                diff = pred[o, k] - gt[o, k]
                dist = np.linalg.norm(diff)
                ade[k] += dist

            fde[K-1] += np.linalg.norm(pred[o, K-1] - gt[o, K-1])

        count += num_obj

    if count > 0:
        ade /= count
        fde[K-1] /= count

    return ade, fde


# --------------------------------------------------
# Evaluation Loop
# --------------------------------------------------

horizon_iou = np.zeros(FUTURE_FRAMES)
horizon_ade = np.zeros(FUTURE_FRAMES)
horizon_fde = np.zeros(FUTURE_FRAMES)

batches = 0

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

        ious = compute_iou_per_horizon(rendered, bev_future)
        ade, fde = compute_motion_errors(motion_pred, motion_target)

        horizon_iou += np.array(ious)
        horizon_ade += ade
        horizon_fde += fde

        batches += 1


horizon_iou /= batches
horizon_ade /= batches
horizon_fde /= batches


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\nLong-Horizon Stability Results")
for k in range(FUTURE_FRAMES):
    print(f"Horizon +{k+1} | IoU: {horizon_iou[k]:.4f} | ADE: {horizon_ade[k]:.4f}")

print(f"FDE@{FUTURE_FRAMES}: {horizon_fde[FUTURE_FRAMES-1]:.4f}")


# --------------------------------------------------
# Plot Degradation Curves
# --------------------------------------------------

horizons = np.arange(1, FUTURE_FRAMES + 1)

plt.figure()
plt.plot(horizons, horizon_iou)
plt.xlabel("Horizon")
plt.ylabel("IoU")
plt.title("IoU Degradation Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "iou_degradation.png"))
plt.close()

plt.figure()
plt.plot(horizons, horizon_ade)
plt.xlabel("Horizon")
plt.ylabel("ADE")
plt.title("ADE Growth Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "ade_growth.png"))
plt.close()

print("\nStability plots saved to outputs/stability/")