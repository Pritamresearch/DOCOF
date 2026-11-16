import os
import matplotlib.pyplot as plt
import numpy as np
import torch
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

FUTURE_FRAMES = 4          # MUST match training
GRID_SIZE = 200
MAX_SAVE_PER_HORIZON = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load nuScenes + official split
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
    past_frames=5,
    future_frames=FUTURE_FRAMES
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda b: {
        "bev_past": torch.stack([x["bev_past"] for x in b]),
        "bev_future": torch.stack([x["bev_future"] for x in b]),
        "boxes": [x["boxes"] for x in b]
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
# Create Output Folders
# --------------------------------------------------

BASE_OUTPUT_DIR = os.path.join("outputs", "object_motion")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

for k in range(FUTURE_FRAMES):
    os.makedirs(
        os.path.join(BASE_OUTPUT_DIR, f"horizon_{k+1}"),
        exist_ok=True
    )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def compute_iou_per_horizon(pred, target):

    K = pred.shape[1]
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    results = []

    for k in range(K):
        p = pred[:, k]
        t = target[:, k]

        TP = (p * t).sum()
        FP = (p * (1 - t)).sum()
        FN = ((1 - p) * t).sum()

        iou = TP / (TP + FP + FN + 1e-6)
        results.append(iou)

    return results


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def save_visualization(gt, pred, horizon, idx):

    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    overlay = np.zeros((*gt.shape, 3))
    overlay[..., 1] = gt
    overlay[..., 0] = pred

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    save_path = os.path.join(
        BASE_OUTPUT_DIR,
        f"horizon_{horizon}",
        f"sample_{idx:04d}.png"
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --------------------------------------------------
# Evaluation + Save Images
# --------------------------------------------------

with torch.no_grad():

    horizon_ious = torch.zeros(FUTURE_FRAMES, device=device)
    saved_per_horizon = [0] * FUTURE_FRAMES
    count = 0

    for batch in loader:

        bev_past = batch["bev_past"].to(device)
        bev_future = batch["bev_future"].to(device)
        boxes = batch["boxes"]

        intent_pred, motion_pred = model(bev_past, boxes)

        rendered = render_future_from_motion(
            boxes,
            motion_pred,
            grid_size=GRID_SIZE,
            future_frames=FUTURE_FRAMES
        ).to(device)

        pred_bin = (torch.sigmoid(rendered) > 0.5).float()

        ious = compute_iou_per_horizon(rendered, bev_future)

        for k in range(FUTURE_FRAMES):
            horizon_ious[k] += ious[k]

        # ---- Save images ----
        for b in range(bev_future.shape[0]):

            for k in range(FUTURE_FRAMES):

                if saved_per_horizon[k] >= MAX_SAVE_PER_HORIZON:
                    continue

                gt = bev_future[b, k, 0]
                pred = pred_bin[b, k, 0]

                if gt.sum() == 0:
                    continue

                save_visualization(
                    gt,
                    pred,
                    horizon=k+1,
                    idx=saved_per_horizon[k]
                )

                saved_per_horizon[k] += 1

        count += 1

        if all(x >= MAX_SAVE_PER_HORIZON for x in saved_per_horizon):
            break

    horizon_ious /= count

    print("\nEvaluation Results")
    for k in range(FUTURE_FRAMES):
        print(f"Horizon +{k+1} IoU: {horizon_ious[k]:.4f}")

    print(f"\nMean IoU: {horizon_ious.mean():.4f}")