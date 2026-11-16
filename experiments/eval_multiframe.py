import os
import torch
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from data.loader import NuScenesIntentDataset
from models.baseline_forecast_v2 import BaselineForecastV2
from models.intent_conditioned_model_v2 import IntentConditionedModelV2


# --------------------------------------------------
# Config
# --------------------------------------------------

DATAROOT = "D:/Pritam/TS BEV/NuScenes"
CHECKPOINT_DIR = "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
FUTURE_FRAMES = 3


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def compute_metrics(pred, target, threshold):

    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    TP = (pred * target).sum(dim=(2, 3, 4))
    FP = (pred * (1 - target)).sum(dim=(2, 3, 4))
    FN = ((1 - pred) * target).sum(dim=(2, 3, 4))
    TN = ((1 - pred) * (1 - target)).sum(dim=(2, 3, 4))

    IoU = TP / (TP + FP + FN + 1e-6)
    Precision = TP / (TP + FP + 1e-6)
    Recall = TP / (TP + FN + 1e-6)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-6)
    Accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    return IoU, Precision, Recall, F1, Accuracy


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

nusc = NuScenes("v1.0-trainval", DATAROOT, verbose=False)
# --------------------------------------------------
# Extract continuous samples from first scene
# --------------------------------------------------

all_tokens = []

for scene in nusc.scene[:20]:   # SAME number used in training
    current_token = scene["first_sample_token"]
    while current_token != "":
        all_tokens.append(current_token)
        sample = nusc.get("sample", current_token)
        current_token = sample["next"]

print("Total samples:", len(all_tokens))

split_idx = int(0.8 * len(all_tokens))
sample_tokens = all_tokens[split_idx:]   # validation split

print("Validation samples:", len(sample_tokens))


dataset = NuScenesIntentDataset(
    nusc=nusc,
    sample_tokens=sample_tokens,
    past_frames=5,
    future_frames=FUTURE_FRAMES
)

def custom_collate(batch):
    return {
        "bev_past": torch.stack([b["bev_past"] for b in batch]),
        "bev_future": torch.stack([b["bev_future"] for b in batch]),
        "intent_label": [b["intent_label"] for b in batch],
        "boxes": [b["boxes"] for b in batch],
    }

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=custom_collate
)


# --------------------------------------------------
# Load Models
# --------------------------------------------------

baseline = BaselineForecastV2().to(DEVICE)
intent_model = IntentConditionedModelV2().to(DEVICE)

baseline.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "baseline_best.pth"), map_location=DEVICE))
intent_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "intent_best.pth"), map_location=DEVICE))

###############

print("\nBaseline weight check:")
for name, param in baseline.named_parameters():
    print(name, param.mean().item())
    break

print("\nIntent weight check:")
for name, param in intent_model.named_parameters():
    print(name, param.mean().item())
    break

###################

baseline.eval()
intent_model.eval()


# --------------------------------------------------
# Threshold Sweep Evaluation
# --------------------------------------------------

thresholds = [0.05, 0.1, 0.2, 0.3, 0.4]

print("\n==== Threshold Sweep Evaluation ====\n")

with torch.no_grad():








    for thresh in thresholds:

        metrics_base = []
        metrics_intent = []

        for batch in dataloader:

            bev_past = batch["bev_past"].to(DEVICE)
            bev_future = batch["bev_future"].to(DEVICE)
            boxes = batch["boxes"]

            pred_base = baseline(bev_past)
            pred_intent, _, _ = intent_model(bev_past, boxes)

#################
            print("Logits mean:", pred_base.mean().item())
            print("Logits std :", pred_base.std().item())
            print("Logits max :", pred_base.max().item())
            print("Logits min :", pred_base.min().item())

            probs = torch.sigmoid(pred_base)
            print("Prob mean :", probs.mean().item())
            print("Prob max  :", probs.max().item())

            print("GT positive pixels:", bev_future.sum().item())
            print("GT mean :", bev_future.mean().item())

            exit()


###################


            metrics_base.append(compute_metrics(pred_base, bev_future, thresh))
            metrics_intent.append(compute_metrics(pred_intent, bev_future, thresh))

        def stack_metrics(metric_list):
            return [torch.cat(m, dim=0) for m in zip(*metric_list)]

        IoU_b, P_b, R_b, F1_b, Acc_b = stack_metrics(metrics_base)
        IoU_i, P_i, R_i, F1_i, Acc_i = stack_metrics(metrics_intent)

        IoU_b = IoU_b.mean().item()
        IoU_i = IoU_i.mean().item()










        print(f"Threshold {thresh:.2f}")
        print(f"  Baseline Mean IoU: {IoU_b:.6f}")
        print(f"  Intent   Mean IoU: {IoU_i:.6f}")
        print()