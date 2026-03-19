"""
Seg-only loss: Lovász + Dice + BCE.
Без classification loss — всё идёт на сегментацию.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    iou = 1.0 - intersection / union
    if p > 1:
        iou[1:] = iou[1:] - iou[:-1]
    return iou


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    perm = perm.detach()
    gt_sorted = labels[perm].float()
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad.to(errors_sorted.dtype))


class LovaszLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        logits = logits.float()
        targets = targets.float()
        B = logits.shape[0]
        loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        for i in range(B):
            loss = loss + _lovasz_hinge_flat(logits[i].reshape(-1), targets[i].reshape(-1))
        return loss / B


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        tgt_flat = targets.reshape(targets.shape[0], -1).float()
        inter = (probs_flat * tgt_flat).sum(1)
        union = probs_flat.sum(1) + tgt_flat.sum(1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegOnlyLoss(nn.Module):
    """
    L = lovasz_w * Lovász + dice_w * Dice + bce_w * BCE

    Без classification — все градиенты идут на сегментацию.
    BCE добавлен для стабильности на ранних эпохах.
    """

    def __init__(
        self,
        lovasz_weight: float = 0.5,
        dice_weight: float = 0.3,
        bce_weight: float = 0.2,
    ):
        super().__init__()
        self.lovasz = LovaszLoss()
        self.dice = DiceLoss()
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, mask_logits: torch.Tensor, mask_targets: torch.Tensor) -> dict:
        """
        mask_logits:  (B, 1, H, W)
        mask_targets: (B, H, W) float 0/1
        """
        l_lovasz = self.lovasz(mask_logits, mask_targets)
        l_dice = self.dice(mask_logits, mask_targets)

        # BCE с pos_weight для дисбаланса (нефть ~2-5% площади)
        if mask_logits.dim() == 4:
            logits_flat = mask_logits.squeeze(1)
        else:
            logits_flat = mask_logits
        l_bce = F.binary_cross_entropy_with_logits(
            logits_flat, mask_targets.float(),
            pos_weight=torch.tensor(5.0, device=mask_logits.device),
        )

        total = (self.lovasz_weight * l_lovasz +
                 self.dice_weight * l_dice +
                 self.bce_weight * l_bce)

        return {
            "total":  total,
            "lovasz": l_lovasz.detach(),
            "dice":   l_dice.detach(),
            "bce":    l_bce.detach(),
        }
