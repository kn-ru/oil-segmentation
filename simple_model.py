"""
SimpleOilNet: ConvNeXt-Tiny backbone + FPN decoder.

~29M параметров. Вход: (B, 2, 512, 512) — VV+VH.
Выходы: mask_logits (B, 1, 512, 512), class_logits (B, 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import timm


class FPNDecoder(nn.Module):
    """Simple top-down FPN decoder."""

    def __init__(self, encoder_channels: List[int], out_channels: int = 64):
        super().__init__()
        # 1×1 lateral проекции для каждого уровня (от deep к shallow)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for ch in encoder_channels
        ])

        # Refinement conv на каждом уровне после fusion
        self.refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for _ in encoder_channels
        ])

        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: [C0@128, C1@64, C2@32, C3@16]
        returns:  (B, out_channels, 128, 128) — уровень C0
        """
        # Lateral projections
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down: начинаем с самого глубокого, применяем refine сразу
        x = self.refines[-1](laterals[-1])  # C3 @ 16×16
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:],
                              mode="bilinear", align_corners=False)
            x = x + laterals[i]
            x = self.refines[i](x)

        return x  # (B, out_channels, 128, 128)


class SimpleOilNet(nn.Module):
    """
    Простая и стабильная архитектура для SAR oil spill.

    Вход:  (B, 2, 512, 512) — VV+VH
    Выход: mask_logits  (B, 1, 512, 512)
           class_logits (B, 3)
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        fpn_channels: int = 128,
        num_classes: int = 3,
    ):
        super().__init__()

        # Backbone: 2-channel input прямо через timm
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            features_only=True, in_chans=2,
        )
        enc_channels = [info["num_chs"] for info in self.encoder.feature_info.info]

        # FPN decoder
        self.decoder = FPNDecoder(enc_channels, fpn_channels)

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            nn.Conv2d(fpn_channels, 1, 1),
        )

        # Classification head: GAP от последнего encoder уровня
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(enc_channels[-1], 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 2, 512, 512)
        Returns:
            mask_logits:  (B, 1, 512, 512)
            class_logits: (B, num_classes)
        """
        features = self.encoder(x)        # [C0..C3]

        dec = self.decoder(features)      # (B, fpn_ch, 128, 128)

        # Seg: upsample 4× к исходному разрешению
        mask_logits = self.seg_head(dec)
        mask_logits = F.interpolate(mask_logits, scale_factor=4,
                                    mode="bilinear", align_corners=False)

        # Cls: от глубокого C3 (768 ch @ 16×16)
        class_logits = self.cls_head(features[-1])

        return {
            "mask_logits":  mask_logits,
            "class_logits": class_logits,
        }
