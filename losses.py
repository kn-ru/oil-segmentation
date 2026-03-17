"""
Loss functions:
  - BinaryFocalLoss (gamma=2, alpha=0.75) + OHEM
  - DiceLoss
  - BoundaryLoss (distance-map based, для мелких пятен)
  - SegmentationLoss = 0.5 * Focal + 0.3 * Dice + 0.2 * Boundary
  - ClassificationLoss = CrossEntropy(label_smoothing=0.05, class_weights)
  - DeepSupervisionLoss = weighted sum of aux seg losses
  - TotalLoss = L_cls + seg_weight * L_seg + ds_weight * L_ds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# ──────────────────── Binary Focal Loss + OHEM ────────────────────────

class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss с опциональным OHEM.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    OHEM: сортируем пиксели по loss, берём top-k% самых сложных.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 ohem_ratio: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio  # 0 = disabled, 0.7 = top 70%

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        pixel_loss = focal_weight * bce  # (B, 1, H, W)

        if self.ohem_ratio > 0:
            # OHEM: keep only top-k hardest pixels per sample
            B = pixel_loss.shape[0]
            pixel_loss_flat = pixel_loss.reshape(B, -1)
            k = max(1, int(pixel_loss_flat.shape[1] * self.ohem_ratio))
            topk_loss, _ = pixel_loss_flat.topk(k, dim=1)
            return topk_loss.mean()

        return pixel_loss.mean()


# ──────────────────── Dice Loss ────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice Loss."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ──────────────────── Boundary Loss ────────────────────────────────────

class BoundaryLoss(nn.Module):
    """Boundary-aware loss для мелких объектов.

    Вычисляет distance map от границ маски и штрафует предсказания
    пропорционально расстоянию до boundary. Нефть ~2% площади —
    boundary learning критичен.

    Упрощённая реализация: Laplacian edge detection + weighted BCE.
    """

    def __init__(self):
        super().__init__()
        # Laplacian kernel для edge detection
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Извлечь boundary пиксели из бинарной маски.

        Args:
            mask: (B, 1, H, W) float, 0/1

        Returns:
            boundary: (B, 1, H, W) float, 1 на границах
        """
        edges = F.conv2d(mask, self.laplacian, padding=1)
        return (edges.abs() > 0).float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        boundary = self._extract_boundary(targets)

        # Весовая карта: boundary pixels важнее (вес 10), остальные (вес 1)
        weight_map = 1.0 + 9.0 * boundary

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted_bce = (weight_map * bce).sum() / weight_map.sum()

        return weighted_bce


# ──────────────────── Combined Segmentation Loss ───────────────────────

class SegmentationLoss(nn.Module):
    """L_seg = w_focal * Focal + w_dice * Dice + w_boundary * Boundary."""

    def __init__(self, focal_gamma=2.0, focal_alpha=0.75,
                 focal_weight=0.5, dice_weight=0.3, boundary_weight=0.2,
                 ohem_ratio=0.7):
        super().__init__()
        self.focal = BinaryFocalLoss(focal_gamma, focal_alpha, ohem_ratio)
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_focal = self.focal(logits, targets)
        l_dice = self.dice(logits, targets)
        l_boundary = self.boundary(logits, targets)
        return (self.focal_weight * l_focal +
                self.dice_weight * l_dice +
                self.boundary_weight * l_boundary)


# ──────────────────── Total Loss ───────────────────────────────────────

class TotalLoss(nn.Module):
    """
    L_total = L_cls + seg_weight * L_seg + ds_weight * L_ds

    Deep supervision: weighted sum of auxiliary segmentation losses.
    """

    def __init__(
        self,
        seg_weight: float = 0.7,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        focal_weight: float = 0.5,
        dice_weight: float = 0.3,
        boundary_weight: float = 0.2,
        ohem_ratio: float = 0.7,
        label_smoothing: float = 0.05,
        num_classes: int = 3,
        class_weights: Optional[List[float]] = None,
        ds_weight: float = 0.4,
    ):
        super().__init__()

        # Classification loss с class weights
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("cls_weights", w)
        else:
            self.cls_weights = None

        self.label_smoothing = label_smoothing
        self.seg_loss = SegmentationLoss(
            focal_gamma, focal_alpha, focal_weight, dice_weight,
            boundary_weight, ohem_ratio,
        )
        self.seg_weight = seg_weight
        self.ds_weight = ds_weight

        # Для deep supervision используем простой Focal+Dice без boundary
        # (aux outputs на низком разрешении, boundary там шумный)
        self.ds_seg_loss = SegmentationLoss(
            focal_gamma, focal_alpha,
            focal_weight=0.6, dice_weight=0.4, boundary_weight=0.0,
            ohem_ratio=0.0,
        )

    def forward(
        self,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_targets: torch.Tensor,
        aux_logits: Optional[List[torch.Tensor]] = None,
        ds_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            class_logits: (B, num_classes)
            labels: (B,) int
            mask_logits: (B, N, 1, H, W)
            mask_targets: (B, N, H, W)
            aux_logits: list of (B*N, 1, H, W) — deep supervision
            ds_weights: list of float — weight per level
        """
        # Classification (cast weights to match logits dtype for AMP)
        w = self.cls_weights.to(dtype=class_logits.dtype) if self.cls_weights is not None else None
        l_cls = F.cross_entropy(
            class_logits, labels,
            weight=w,
            label_smoothing=self.label_smoothing,
        )

        # Main segmentation
        B, N = mask_logits.shape[:2]
        ml_flat = mask_logits.reshape(B * N, 1, mask_logits.shape[-2], mask_logits.shape[-1])
        mt_flat = mask_targets.reshape(B * N, mask_targets.shape[-2], mask_targets.shape[-1])
        l_seg = self.seg_loss(ml_flat, mt_flat)

        total = l_cls + self.seg_weight * l_seg

        # Deep supervision
        l_ds = torch.tensor(0.0, device=total.device)
        if aux_logits and ds_weights:
            for aux, w in zip(aux_logits, ds_weights):
                # aux: (B*N, 1, H, W), mt_flat already matches after upsampling
                l_aux = self.ds_seg_loss(aux, mt_flat)
                l_ds = l_ds + w * l_aux
            l_ds = l_ds / sum(ds_weights)
            total = total + self.ds_weight * l_ds

        return {
            "total": total,
            "cls": l_cls.detach(),
            "seg": l_seg.detach(),
            "ds": l_ds.detach(),
        }
