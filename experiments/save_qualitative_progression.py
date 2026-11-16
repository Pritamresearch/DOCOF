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


# ==========================
# CONFIG
# ==========================

DATAROOT = "D:/Pritam/TS BEV/NuScenes"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs/qualitative_progression"

FUTURE_FRAMES = 4
PAST_FRAMES = 5
GRID_SIZE = 200
NUM_EXAMPLES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# Dataset
# ==========================

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

loader = DataLoader(dataset, batch_size=1, shuffle=False)


# ==========================
# Model
# ==========================

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


# ==========================
# Save Progression Grid
# ==========================

saved = 0

with torch.no_grad():

    for idx, batch in enumerate(loader):

        if saved >= NUM_EXAMPLES:
            break

        bev_past = batch["bev_past"].to(device)
        bev_future = batch["bev_future"].to(device)
        boxes = batch["boxes"]

        _, motion_pred = model(bev_past, boxes)

        rendered = render_future_from_motion(
            boxes,
            motion_pred,
            grid_size=GRID_SIZE,
            future_frames=FUTURE_FRAMES
        ).to(device)

        rendered = torch.sigmoid(rendered)
        rendered = (rendered > 0.5).float()

        # Convert to numpy
        gt = bev_future[0].cpu().numpy()       # (K,1,H,W)
        pred = rendered[0].cpu().numpy()

        fig, axes = plt.subplots(2, FUTURE_FRAMES, figsize=(12, 5))

        for k in range(FUTURE_FRAMES):

            # Ground Truth
            axes[0, k].imshow(gt[k, 0], cmap="gray")
            axes[0, k].set_title(f"GT t+{k+1}")
            axes[0, k].axis("off")

            # Prediction
            axes[1, k].imshow(pred[k, 0], cmap="gray")
            axes[1, k].set_title(f"Pred t+{k+1}")
            axes[1, k].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"progression_{saved+1}.png"))
        plt.close()

        saved += 1

print(f"Saved {saved} progression grids to {OUTPUT_DIR}")