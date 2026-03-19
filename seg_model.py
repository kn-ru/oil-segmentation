"""
SegOnlyNet: ConvNeXt-V2 + UperNet decoder — SOTA сегментация.

UperNet = PPM (глобальный контекст) + FPN (multi-scale) + Fusion.
Доказано лучший decoder для semantic segmentation (ADE20K, Cityscapes).

~50M параметров (ConvNeXt-V2-Small). Вход: (B, 2, 512, 512).
Выход: mask_logits (B, 1, 512, 512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import timm


class PPM(nn.Module):
    """Pyramid Pooling Module — глобальный контекст на разных масштабах."""

    def __init__(self, in_channels: int, out_channels: int,
                 pool_scales: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ))
        # Bottleneck после concat: in_ch + len(scales)*out_ch → out_ch
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels,
                      out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        ppm_outs = [x]
        for stage in self.stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=(h, w),
                                   mode="bilinear", align_corners=False)
            ppm_outs.append(pooled)
        return self.bottleneck(torch.cat(ppm_outs, dim=1))


class UperNetDecoder(nn.Module):
    """
    UperNet decoder:
    1. PPM на C4 → глобальный контекст
    2. FPN lateral projections на C0..C3
    3. Top-down pathway с concat
    4. FPN fusion: concat всех уровней → финальная карта
    """

    def __init__(self, encoder_channels: List[int], fpn_channels: int = 256,
                 pool_scales: tuple = (1, 2, 3, 6)):
        super().__init__()

        # PPM на самом глубоком уровне
        self.ppm = PPM(encoder_channels[-1], fpn_channels, pool_scales)

        # FPN lateral projections (1×1 conv для каждого encoder уровня)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.GELU(),
            )
            for ch in encoder_channels[:-1]  # C0..C2 (C3 через PPM)
        ])

        # FPN refinement convs
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.GELU(),
            )
            for _ in encoder_channels[:-1]
        ])

        # FPN fusion: concat all levels → reduce
        # len(encoder_channels) уровней × fpn_channels
        self.fpn_fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * len(encoder_channels), fpn_channels,
                      3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
        )

        self.out_channels = fpn_channels

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: [C0@128, C1@64, C2@32, C3@16]
        returns:  (B, fpn_channels, 128, 128)
        """
        # 1. PPM на C3 (deepest) → глобальный контекст
        ppm_out = self.ppm(features[-1])  # (B, fpn_ch, 16, 16)

        # 2. FPN lateral projections для C0..C2
        laterals = [lat(f) for lat, f in zip(self.laterals, features[:-1])]

        # 3. Top-down pathway: PPM → C2 → C1 → C0
        fpn_outs = []

        # Начинаем с PPM output
        x = ppm_out
        for i in range(len(laterals) - 1, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:],
                              mode="bilinear", align_corners=False)
            x = x + laterals[i]
            x = self.fpn_convs[i](x)
            fpn_outs.insert(0, x)

        # Добавляем PPM output как самый глубокий уровень
        fpn_outs.append(ppm_out)

        # 4. FPN fusion: upsample все до размера C0, concat
        target_size = fpn_outs[0].shape[-2:]
        upsampled = []
        for out in fpn_outs:
            if out.shape[-2:] != target_size:
                out = F.interpolate(out, size=target_size,
                                    mode="bilinear", align_corners=False)
            upsampled.append(out)

        fused = self.fpn_fusion(torch.cat(upsampled, dim=1))
        return fused


class SegOnlyNet(nn.Module):
    """
    ConvNeXt-V2 encoder + UperNet decoder — чистая сегментация.

    Архитектура:
        ConvNeXt-V2-Tiny → [96, 192, 384, 768] @ [128, 64, 32, 16]
        UperNet decoder:
            PPM(C3) → глобальный контекст
            FPN(C0..C3) → multi-scale
            Fusion → concat всех уровней
        Seg head: 2× Conv3x3 + 1×1 → binary mask
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        fpn_channels: int = 256,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            features_only=True, in_chans=2,
        )
        enc_channels = [info["num_chs"] for info in self.encoder.feature_info.info]

        self.decoder = UperNetDecoder(enc_channels, fpn_channels)

        # Segmentation head: мощнее для лучших границ
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.GELU(),
            nn.Conv2d(fpn_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        dec = self.decoder(features)      # (B, fpn_ch, 128, 128)
        mask_logits = self.seg_head(dec)
        mask_logits = F.interpolate(mask_logits, scale_factor=4,
                                    mode="bilinear", align_corners=False)
        return mask_logits
