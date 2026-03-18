"""
Loss для SimpleOilNet:
  L_seg = lovasz_weight * LovászLoss + dice_weight * DiceLoss
  L_cls = CrossEntropy(class_weights, label_smoothing)
  L     = L_cls + seg_weight * L_seg

Lovász loss напрямую оптимизирует IoU — лучший выбор для сегментации
маленьких объектов (~2% площади).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ──────────────────── Lovász Loss ──────────────────────────────────────
# Реализация: https://arxiv.org/abs/1705.08790

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Вычислить веса для Lovász расширения."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    iou = 1.0 - intersection / union
    if p > 1:
        iou[1:] = iou[1:] - iou[:-1]
    return iou


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovász hinge loss для плоских тензоров."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    perm = perm.detach()
    gt_sorted = labels[perm].float()
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad.to(errors_sorted.dtype))
    return loss


class LovaszLoss(nn.Module):
    """Binary Lovász-Hinge Loss для сегментации.

    Непосредственно оптимизирует IoU — лучше Dice для маленьких объектов.
    Принимает логиты (без sigmoid).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, H, W) или (B, H, W)
            targets: (B, H, W) float, 0/1
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        # Compute in float32 for numerical stability (AMP может дать float16)
        logits = logits.float()
        targets = targets.float()

        B = logits.shape[0]
        loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        for i in range(B):
            flat_log = logits[i].reshape(-1)
            flat_lab = targets[i].reshape(-1)
            loss = loss + _lovasz_hinge_flat(flat_log, flat_lab)

        return loss / B


# ──────────────────── Dice Loss ────────────────────────────────────────

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


# ──────────────────── Combined Loss ────────────────────────────────────

class SimpleLoss(nn.Module):
    """
    L = L_cls + seg_weight * (lovasz_w * Lovász + dice_w * Dice)
    """

    def __init__(
        self,
        seg_weight: float = 2.0,
        lovasz_weight: float = 0.6,
        dice_weight: float = 0.4,
        label_smoothing: float = 0.05,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.lovasz = LovaszLoss()
        self.dice = DiceLoss()
        self.seg_weight = seg_weight
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("cls_weights", w)
        else:
            self.cls_weights = None

    def forward(
        self,
        class_logits: torch.Tensor,  # (B, 3)
        labels: torch.Tensor,        # (B,)
        mask_logits: torch.Tensor,   # (B, 1, H, W)
        mask_targets: torch.Tensor,  # (B, H, W)
    ) -> dict:
        # Classification
        w = self.cls_weights.to(class_logits.dtype) if self.cls_weights is not None else None
        l_cls = F.cross_entropy(class_logits, labels,
                                weight=w, label_smoothing=self.label_smoothing)

        # Segmentation
        l_lovasz = self.lovasz(mask_logits, mask_targets)
        l_dice = self.dice(mask_logits, mask_targets)
        l_seg = self.lovasz_weight * l_lovasz + self.dice_weight * l_dice

        total = l_cls + self.seg_weight * l_seg

        return {
            "total":   total,
            "cls":     l_cls.detach(),
            "seg":     l_seg.detach(),
            "lovasz":  l_lovasz.detach(),
            "dice":    l_dice.detach(),
        }
