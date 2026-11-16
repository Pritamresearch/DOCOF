import torch
import torch.nn.functional as F


def masked_l2(pred, gt, mask):

    diff = pred - gt
    error = torch.sqrt((diff * diff).sum(dim=-1) + 1e-6)

    error = error * mask  # NOT inplace

    return error.sum() / (mask.sum() + 1e-6)


def ade_loss_v2(pred, gt, masks):
    return sum(
        masked_l2(p[:, :, :2], g[:, :, :2], masks[k])
        for k, (p, g) in enumerate(zip(pred, gt))
    ) / len(pred)


def fde_loss_v2(pred, gt, masks):
    return masked_l2(
        pred[-1][:, :, :2],
        gt[-1][:, :, :2],
        masks[-1]
    )


def velocity_loss_v2(pred, gt, masks):
    return sum(
        masked_l2(p[:, :, 2:4], g[:, :, 2:4], masks[k])
        for k, (p, g) in enumerate(zip(pred, gt))
    ) / len(pred)


def yaw_loss_v2(pred, gt, masks):
    loss = 0
    for k, (p, g) in enumerate(zip(pred, gt)):
        mask = masks[k]
        pred_vec = torch.stack(
            [torch.sin(p[:, :, 4]), torch.cos(p[:, :, 4])], dim=-1
        )
        gt_vec = torch.stack(
            [torch.sin(g[:, :, 4]), torch.cos(g[:, :, 4])], dim=-1
        )
        loss += masked_l2(pred_vec, gt_vec, mask)
    return loss / len(pred)


def occupancy_loss_v2(pred, gt):

    pos_weight = 50.0

    weight = torch.ones_like(gt)
    weight[gt > 0.5] = pos_weight

    return torch.nn.functional.binary_cross_entropy(
        pred,
        gt,
        weight=weight
    )

def control_regularization_v2(controls):
    return sum(torch.mean(u**2) for u in controls) / len(controls)