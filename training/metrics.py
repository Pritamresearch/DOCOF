import torch
import numpy as np


def compute_iou(pred, gt, threshold=0.5):
    """
    Compute Intersection-over-Union between predicted probabilities and binary ground-truth.
    """
    pred_binary = (pred > threshold)
    gt_binary = (gt > 0.5)

    intersection = (pred_binary & gt_binary).float().sum()
    union = (pred_binary | gt_binary).float().sum()

    return (intersection / (union + 1e-6)).item()


def compute_ade(pred_positions, gt_positions, masks=None):
    """
    Average Displacement Error (ADE) across horizons.
    """
    errors = []
    for k in range(len(pred_positions)):
        diff = pred_positions[k][..., :2] - gt_positions[k][..., :2]
        dist = torch.norm(diff, dim=-1)
        if masks is not None:
            mask = masks[k].bool()
            if mask.sum() > 0:
                errors.append(dist[mask].mean().item())
        else:
            errors.append(dist.mean().item())
    return float(np.mean(errors)) if errors else 0.0


def compute_fde(pred_final, gt_final, mask=None):
    """
    Final Displacement Error (FDE) at horizon K.
    """
    diff = pred_final[..., :2] - gt_final[..., :2]
    dist = torch.norm(diff, dim=-1)
    if mask is not None:
        mask = mask.bool()
        if mask.sum() > 0:
            return float(dist[mask].mean().item())
    return float(dist.mean().item())


def compute_smooth_std(errors_per_horizon):
    """
    Sample standard deviation of per-horizon displacement errors.
    Lower values indicate smoother, non-divergent degradation.
    """
    return float(np.std(errors_per_horizon, ddof=1)) if len(errors_per_horizon) > 1 else 0.0


def compute_occupancy_mass(occupancy_grid):
    """
    Sum of occupied-cell probabilities over the BEV grid.
    """
    return float(occupancy_grid.sum().item())
