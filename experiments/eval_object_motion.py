import os
import torch
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from data.loader import NuScenesIntentDataset
from models.object_motion_forecast import ObjectMotionForecast
from utils.render_future import render_future_from_motion


DATAROOT = "D:/Pritam/TS BEV/NuScenes"
CHECKPOINT_DIR = "checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nusc = NuScenes("v1.0-trainval", DATAROOT, verbose=False)

# ============================================================
# LOAD SAME 30 SCENES AS TRAINING
# ============================================================

NUM_SCENES = 30

all_tokens = []

for scene in nusc.scene[:NUM_SCENES]:
    current_token = scene["first_sample_token"]
    while current_token != "":
        all_tokens.append(current_token)
        sample = nusc.get("sample", current_token)
        current_token = sample["next"]

split_idx = int(0.8 * len(all_tokens))
val_tokens = all_tokens[split_idx:]

dataset = NuScenesIntentDataset(
    nusc=nusc,
    sample_tokens=val_tokens,
    past_frames=5,
    future_frames=3
)


def custom_collate(batch):
    return {
        "bev_past": torch.stack([b["bev_past"] for b in batch]),
        "bev_future": torch.stack([b["bev_future"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
    }


dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=custom_collate)

# Detect correct intent class count
max_label = 0
for i in range(min(20, len(dataset))):
    sample = dataset[i]
    if sample["intent_label"].numel() > 0:
        max_label = max(max_label, sample["intent_label"].max().item())

NUM_INTENT_CLASSES = int(max_label + 1)
print("Detected intent classes:", NUM_INTENT_CLASSES)

model = ObjectMotionForecast(
    future_frames=3,
    num_intent_classes=NUM_INTENT_CLASSES
).to(device)


model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "object_motion_best.pth"),
                                 map_location=device))
model.eval()


def compute_iou(pred, target):

    pred = (pred > 0.5).float()

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    return TP / (TP + FP + FN + 1e-6)


with torch.no_grad():

    total_iou = 0.0
    count = 0

    for batch in dataloader:

        bev_past = batch["bev_past"].to(device)
        bev_future = batch["bev_future"].to(device)
        boxes = batch["boxes"]

        intent_pred, motion_pred = model(bev_past, boxes)

        rendered = render_future_from_motion(
            boxes, motion_pred, grid_size=200, future_frames=3
        ).to(device)

        iou = compute_iou(rendered, bev_future)

        total_iou += iou.item()
        count += 1

    print("Mean IoU:", total_iou / count)