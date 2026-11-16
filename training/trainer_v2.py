import torch
import csv
import os
from tqdm import tqdm

from losses.losses_v2 import (
    ade_loss_v2,
    fde_loss_v2,
    velocity_loss_v2,
    yaw_loss_v2,
    occupancy_loss_v2,
    control_regularization_v2,
    masked_l2
)


# ---------------------------------------------------------
# IoU computation (logits input)
# ---------------------------------------------------------

def compute_iou(pred, gt, threshold=0.3):

    pred = pred > threshold
    gt = gt > 0.5

    intersection = (pred & gt).float().sum()
    union = (pred | gt).float().sum()

    return (intersection / (union + 1e-6)).item()


# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

class TrainerV2:

    def __init__(self, model, optimizer, cfg, device):

        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device

        self.future_frames = cfg["future_frames"]
        self.grad_clip = cfg["grad_clip"]

        # Best trackers
        self.best_fde_k = [float("inf")] * self.future_frames
        self.best_mean = float("inf")
        self.best_smooth = float("inf")

        # CSV
        self.csv_path = "training_log_v2.csv"

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)

                header = (
                    ["epoch", "train_loss"]
                    + [f"fde_t{k+1}" for k in range(self.future_frames)]
                    + [f"iou_t{k+1}" for k in range(self.future_frames)]
                    + ["mean_fde", "mean_iou", "smooth_std"]
                )

                writer.writerow(header)

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    def compute_loss(self,
                     states_pred,
                     occ_pred,
                     controls,
                     future_gt,
                     occ_gt,
                     validity_masks):

        # Occupancy loss
        loss_occ = 0
        for k in range(self.future_frames):
            loss_occ += occupancy_loss_v2(
                occ_pred[k],
                occ_gt[k]
            )
        loss_occ /= self.future_frames

        loss_ade = ade_loss_v2(states_pred, future_gt, validity_masks)
        loss_fde = fde_loss_v2(states_pred, future_gt, validity_masks)
        loss_vel = velocity_loss_v2(states_pred, future_gt, validity_masks)
        loss_yaw = yaw_loss_v2(states_pred, future_gt, validity_masks)
        loss_reg = control_regularization_v2(controls)

        total = (
            loss_occ
            + 0.5 * loss_ade
            + 0.5 * loss_fde
            + 0.3 * loss_vel
            + 0.2 * loss_yaw
            + 0.01 * loss_reg
        )

        return total

    # ---------------------------------------------------------
    # Training Epoch
    # ---------------------------------------------------------

    def train_epoch(self, loader):

        self.model.train()
        total_loss = 0

        for batch in tqdm(loader, desc="Training"):

            history = batch["history"].to(self.device)
            state_t = batch["state_t"].to(self.device)
            future_gt = [f.to(self.device) for f in batch["future_states"]]
            occ_gt = [o.to(self.device) for o in batch["future_occupancy"]]
            validity_masks = [
                m.to(self.device) for m in batch["validity_masks"]
            ]

            states_pred, occ_pred, controls = self.model(
                history,
                state_t,
                validity_masks
            )

            loss = self.compute_loss(
                states_pred,
                occ_pred,
                controls,
                future_gt,
                occ_gt,
                validity_masks
            )

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate(self, loader):

        self.model.eval()

        fde_k = torch.zeros(self.future_frames)
        iou_k = torch.zeros(self.future_frames)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):

                history = batch["history"].to(self.device)
                state_t = batch["state_t"].to(self.device)
                future_gt = [f.to(self.device) for f in batch["future_states"]]
                occ_gt = [o.to(self.device) for o in batch["future_occupancy"]]
                validity_masks = [
                    m.to(self.device) for m in batch["validity_masks"]
                ]

                states_pred, occ_pred, controls = self.model(
                    history,
                    state_t,
                    validity_masks
                )

                for k in range(self.future_frames):

                    # FDE@k
                    fde_k[k] += masked_l2(
                        states_pred[k][:, :, :2],
                        future_gt[k][:, :, :2],
                        validity_masks[k]
                    ).item()

                    # IoU@k
                    iou_k[k] += compute_iou(
                        occ_pred[k],
                        occ_gt[k]
                    )

        fde_k /= len(loader)
        iou_k /= len(loader)

        mean_fde = fde_k.mean().item()
        mean_iou = iou_k.mean().item()
        smooth_std = fde_k.std().item()

        return (
            fde_k.tolist(),
            iou_k.tolist(),
            mean_fde,
            mean_iou,
            smooth_std
        )

    # ---------------------------------------------------------
    # Fit Loop
    # ---------------------------------------------------------

    def fit(self, train_loader, val_loader, epochs, save_path):

        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)

            fde_k, iou_k, mean_fde, mean_iou, smooth_std = \
                self.validate(val_loader)

            print(f"\nEpoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")

            for k in range(self.future_frames):
                print(
                    f"FDE@t+{k+1}: {fde_k[k]:.4f} | "
                    f"IoU@t+{k+1}: {iou_k[k]:.4f}"
                )

            print(f"Mean FDE: {mean_fde:.4f}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"Smooth STD (FDE): {smooth_std:.4f}")

            # --------------------------------------
            # Save Best Horizon-wise Models
            # --------------------------------------

            for k in range(self.future_frames):
                if fde_k[k] < self.best_fde_k[k]:
                    self.best_fde_k[k] = fde_k[k]
                    torch.save(
                        self.model.state_dict(),
                        f"docof_v2_best_t{k+1}.pth"
                    )

            # --------------------------------------
            # Save Best Mean Performer
            # --------------------------------------

            if mean_fde < self.best_mean:
                self.best_mean = mean_fde
                torch.save(
                    self.model.state_dict(),
                    "docof_v2_best_mean.pth"
                )

            # --------------------------------------
            # Save Best Smooth Degradation
            # --------------------------------------

            if smooth_std < self.best_smooth:
                self.best_smooth = smooth_std
                torch.save(
                    self.model.state_dict(),
                    "docof_v2_best_smooth.pth"
                )

            # --------------------------------------
            # CSV Logging
            # --------------------------------------

            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)

                writer.writerow(
                    [epoch, train_loss]
                    + fde_k
                    + iou_k
                    + [mean_fde, mean_iou, smooth_std]
                )