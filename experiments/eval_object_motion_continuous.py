import os
import torch
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from data.loader import NuScenesIntentDataset
from models.object_motion_forecast_continuous import ObjectMotionForecastContinuous
from utils.render_future_continuous import render_future_from_motion_continuous


DATAROOT = "D:/Pritam/TS BEV/NuScenes"
CHECKPOINT_DIR = "checkpoints"

GRID_SIZE = 200
FUTURE_FRAMES = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nusc = NuScenes("v1.0-trainval", DATAROOT, verbose=False)

scene = nusc.scene[0]
token = scene["first_sample_token"]

tokens = []
while token != "":
    tokens.append(token)
    sample = nusc.get("sample", token)
    token = sample["next"]

split = int(0.8 * len(tokens))
val_tokens = tokens[split:]


dataset = NuScenesIntentDataset(
    nusc=nusc,
    sample_tokens=val_tokens,
    past_frames=5,
    future_frames=FUTURE_FRAMES
)


def custom_collate(batch):
    return {
        "bev_past": torch.stack([b["bev_past"] for b in batch]),
        "bev_future": torch.stack([b["bev_future"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
    }


loader = DataLoader(dataset, batch_size=4,
                    shuffle=False, collate_fn=custom_collate)

# detect intent classes
max_label = 0
for i in range(min(50, len(dataset))):
    sample = dataset[i]
    if sample["intent_label"].numel() > 0:
        max_label = max(max_label, sample["intent_label"].max().item())

NUM_INTENT_CLASSES = int(max_label + 1)

model = ObjectMotionForecastContinuous(
    future_frames=FUTURE_FRAMES,
    num_intent_classes=NUM_INTENT_CLASSES
).to(device)

model.load_state_dict(torch.load(
    os.path.join(CHECKPOINT_DIR,
                 "object_motion_continuous_best.pth"),
    map_location=device
))

model.eval()


def compute_iou(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    return TP / (TP + FP + FN + 1e-6)


with torch.no_grad():

    total_iou = 0.0
    count = 0

    for batch in loader:

        bev_past = batch["bev_past"].to(device)
        bev_future = batch["bev_future"].to(device)
        boxes = batch["boxes"]

        intent_pred, motion_pred = model(bev_past, boxes)

        rendered = render_future_from_motion_continuous(
            boxes, motion_pred,
            grid_size=GRID_SIZE,
            future_frames=FUTURE_FRAMES
        )

        iou = compute_iou(rendered, bev_future)

        total_iou += iou.item()
        count += 1

    print("Mean IoU (Continuous):", total_iou / count)