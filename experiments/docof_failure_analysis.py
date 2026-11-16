import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config_v2 import CONFIG_V2
from models.docof_model_v2 import DOCOFv2
from data.docof_torch_dataset_v2 import DOCOFTorchDatasetV2


# -------------------------------------------------------
# IoU
# -------------------------------------------------------

def compute_iou(pred, gt, threshold=0.3):

    pred_bin = pred > threshold
    gt_bin = gt > 0.5

    intersection = (pred_bin & gt_bin).float().sum()
    union = (pred_bin | gt_bin).float().sum()

    return (intersection / (union + 1e-6)).item()


# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

def save_visualization(pred, gt, save_path):

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    fig, axs = plt.subplots(1,3, figsize=(12,4))

    axs[0].imshow(gt, cmap="gray")
    axs[0].set_title("Ground Truth")

    axs[1].imshow(pred, cmap="hot")
    axs[1].set_title("Prediction")

    overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
    overlay[...,0] = pred
    overlay[...,1] = gt

    axs[2].imshow(overlay)
    axs[2].set_title("Overlay")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DOCOFTorchDatasetV2(
        root="preprocessed_docof_v2",
        split="val"
    )

    model = DOCOFv2(CONFIG_V2).to(device)

    model.load_state_dict(
        torch.load("docof_v2_best_mean.pth", map_location=device)
    )

    model.eval()

    failure_cases = []

    print("Running inference...")

    with torch.no_grad():

        for idx in tqdm(range(len(dataset))):

            sample = dataset[idx]

            history = sample["history"].unsqueeze(0).to(device)
            state_t = sample["state_t"].unsqueeze(0).to(device)

            masks = [
                m.unsqueeze(0).to(device)
                for m in sample["validity_masks"]
            ]

            occ_gt = [
                o.to(device)
                for o in sample["future_occupancy"]
            ]

            states_pred, occ_pred, _ = model(
                history,
                state_t,
                masks
            )

            # Evaluate last horizon
            pred = occ_pred[-1].squeeze(0)
            gt = occ_gt[-1]

            iou = compute_iou(pred, gt)

            failure_cases.append((iou, idx, pred.cpu(), gt.cpu()))

    # Sort by worst IoU
    failure_cases.sort(key=lambda x: x[0])

    save_dir = "failure_cases"
    os.makedirs(save_dir, exist_ok=True)

    print("Saving worst predictions...")

    for rank, (iou, idx, pred, gt) in enumerate(failure_cases[:10]):

        save_path = os.path.join(
            save_dir,
            f"failure_{rank}_iou_{iou:.3f}.png"
        )

        save_visualization(pred, gt, save_path)

    print("Failure cases saved in:", save_dir)


if __name__ == "__main__":
    main()