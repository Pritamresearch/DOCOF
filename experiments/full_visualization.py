import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
OUTPUT_DIR = "outputs/full_visualization"

FUTURE_FRAMES = 4
PAST_FRAMES = 5
GRID_SIZE = 200
NUM_EXAMPLES = 3

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
# Helper: Load RGB image
# ==========================

def load_cam_front(sample_token):
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)
    img_path = os.path.join(DATAROOT, cam_data["filename"])
    img = Image.open(img_path).convert("RGB")
    return np.array(img)


# ==========================
# Run Visualization
# ==========================

saved = 0

with torch.no_grad():

    for idx, batch in enumerate(loader):

        if saved >= NUM_EXAMPLES:
            break

        sample_token = val_tokens[idx]
        sample = nusc.get("sample", sample_token)

        bev_past = batch["bev_past"].to(device)
        bev_future = batch["bev_future"].to(device)
        boxes = batch["boxes"]

        # Present raw image
        raw_present = load_cam_front(sample_token)

        # Present GT BEV
        present_gt_bev = bev_past[0, -1, 0].cpu().numpy()

        # Predict future
        _, motion_pred = model(bev_past, boxes)

        rendered = render_future_from_motion(
            boxes,
            motion_pred,
            grid_size=GRID_SIZE,
            future_frames=FUTURE_FRAMES
        ).to(device)

        rendered = torch.sigmoid(rendered)
        rendered = (rendered > 0.5).float()

        pred_bev = rendered[0].cpu().numpy()
        gt_bev = bev_future[0].cpu().numpy()

        # Collect future raw images
        future_raw_images = []
        next_token = sample["next"]

        for k in range(FUTURE_FRAMES):
            if next_token == "":
                break

            future_raw_images.append(load_cam_front(next_token))
            next_sample = nusc.get("sample", next_token)
            next_token = next_sample["next"]

        # ==========================
        # Plot Grid
        # ==========================

        rows = 3
        cols = FUTURE_FRAMES + 1

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

        # Row 1: Raw images
        axes[0,0].imshow(raw_present)
        axes[0,0].set_title("Present RGB")
        axes[0,0].axis("off")

        for k in range(FUTURE_FRAMES):
            if k < len(future_raw_images):
                axes[0,k+1].imshow(future_raw_images[k])
                axes[0,k+1].set_title(f"Future RGB t+{k+1}")
            axes[0,k+1].axis("off")

        # Row 2: GT BEV
        axes[1,0].imshow(present_gt_bev, cmap="gray")
        axes[1,0].set_title("Present GT BEV")
        axes[1,0].axis("off")

        for k in range(FUTURE_FRAMES):
            axes[1,k+1].imshow(gt_bev[k,0], cmap="gray")
            axes[1,k+1].set_title(f"GT BEV t+{k+1}")
            axes[1,k+1].axis("off")

        # Row 3: Predicted BEV
        axes[2,0].axis("off")

        for k in range(FUTURE_FRAMES):
            axes[2,k+1].imshow(pred_bev[k,0], cmap="gray")
            axes[2,k+1].set_title(f"Pred BEV t+{k+1}")
            axes[2,k+1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"full_vis_{saved+1}.png"))
        plt.close()

        saved += 1

print(f"Saved {saved} full visualizations to {OUTPUT_DIR}")