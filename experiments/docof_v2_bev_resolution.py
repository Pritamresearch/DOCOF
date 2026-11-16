import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from config_v2 import CONFIG_V2
from models.docof_model_v2 import DOCOFv2
from data.docof_torch_dataset_v2 import DOCOFTorchDatasetV2


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

def compute_fde(pred, gt, mask):

    error = torch.norm(pred[:, :, :2] - gt[:, :, :2], dim=-1)
    error = error * mask

    return error.sum() / (mask.sum() + 1e-6)


def compute_iou(pred, gt, threshold=0.3):

    pred_bin = pred > threshold
    gt_bin = gt > 0.5

    intersection = (pred_bin & gt_bin).float().sum()
    union = (pred_bin | gt_bin).float().sum()

    return (intersection / (union + 1e-6)).item()


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate_grid_resolution(grid_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CONFIG_V2["grid_size"] = grid_size

    print(f"\nEvaluating BEV grid resolution: {grid_size}x{grid_size}")

    dataset = DOCOFTorchDatasetV2(
        root="preprocessed_docof_v2",
        split="val"
    )

    model = DOCOFv2(CONFIG_V2).to(device)

    checkpoint = torch.load(
        "docof_v2_best_mean.pth",
        map_location=device
    )

    model_dict = model.state_dict()

    filtered_checkpoint = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(filtered_checkpoint)
    model.load_state_dict(model_dict)

    model.eval()

    future_frames = CONFIG_V2["future_frames"]

    fde_k = torch.zeros(future_frames)
    iou_k = torch.zeros(future_frames)

    with torch.no_grad():

        for idx in tqdm(range(len(dataset)), desc=f"Grid {grid_size}"):

            sample = dataset[idx]

            history = sample["history"].unsqueeze(0).to(device)
            state_t = sample["state_t"].unsqueeze(0).to(device)

            future_gt = [
                f.unsqueeze(0).to(device)
                for f in sample["future_states"]
            ]

            occ_gt = [
                o.to(device)
                for o in sample["future_occupancy"]
            ]

            masks = [
                m.unsqueeze(0).to(device)
                for m in sample["validity_masks"]
            ]

            states_pred, occ_pred, _ = model(
                history,
                state_t,
                masks
            )

            for k in range(future_frames):

                # FDE
                fde_k[k] += compute_fde(
                    states_pred[k],
                    future_gt[k],
                    masks[k]
                ).item()

                # resize GT occupancy to grid size
                gt_occ = occ_gt[k].unsqueeze(0).unsqueeze(0)

                gt_occ = F.interpolate(
                    gt_occ.float(),
                    size=(grid_size, grid_size),
                    mode="nearest"
                ).squeeze()

                pred_occ = occ_pred[k].squeeze(0)

                iou_k[k] += compute_iou(
                    pred_occ,
                    gt_occ,
                    threshold=0.3
                )

    fde_k /= len(dataset)
    iou_k /= len(dataset)

    return fde_k, iou_k


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    grid_sizes = [128, 200, 256]

    results = {}

    for grid in grid_sizes:

        fde_k, iou_k = evaluate_grid_resolution(grid)

        results[grid] = {
            "fde": fde_k,
            "iou": iou_k
        }

    print("\n==============================")
    print("DOCOF GRID + HORIZON RESULTS")
    print("==============================")

    future_frames = CONFIG_V2["future_frames"]

    for grid in grid_sizes:

        print(f"\nGrid {grid}x{grid}")

        for k in range(future_frames):

            print(
                f"t+{k+1} | "
                f"FDE: {results[grid]['fde'][k]:.4f} | "
                f"IoU: {results[grid]['iou'][k]:.4f}"
            )

    print("\n----- Mean Results -----")

    for grid in grid_sizes:

        print(
            f"{grid}x{grid} | "
            f"Mean IoU: {results[grid]['iou'].mean():.4f} | "
            f"Mean FDE: {results[grid]['fde'].mean():.4f}"
        )


if __name__ == "__main__":
    main()