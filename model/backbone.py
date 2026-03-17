"""
ConvNeXt backbone с адаптацией входных каналов.

Используем ConvNeXt-Small (preferred) или ConvNeXt-Tiny (fallback).
Channels: [96, 192, 384, 768] для обоих вариантов.

Backbone получает фьюженные фичи (96ch) после PolarimetricFusion.
Поскольку fusion output уже на stride 4 (128x128 для входа 512),
а ConvNeXt downstem даёт stride 4, заменяем downstem на identity/1x1 conv.
"""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def build_convnext_backbone(
    name: str = "convnext_small",
    in_channels: int = 96,
    pretrained: bool = True,
) -> nn.Module:
    """Строит ConvNeXt backbone с кастомным входом.

    Returns:
        ConvNeXtBackbone с атрибутом .stages и .channels
    """
    if not HAS_TIMM:
        raise ImportError("timm is required for ConvNeXt backbone: pip install timm")

    # Пробуем Small, fallback на Tiny
    try:
        model = timm.create_model(name, pretrained=pretrained, features_only=True)
    except Exception:
        print(f"[!] {name} not available, falling back to convnext_tiny")
        model = timm.create_model("convnext_tiny", pretrained=pretrained,
                                  features_only=True)

    return ConvNeXtBackbone(model, in_channels)


class ConvNeXtBackbone(nn.Module):
    """Обёртка над timm ConvNeXt features_only.

    Заменяет stem (stride 4 patchify) на 1x1 conv адаптер,
    так как наш вход уже на stride 4 от оригинального разрешения.
    """

    def __init__(self, timm_model: nn.Module, in_channels: int = 96):
        super().__init__()
        self.model = timm_model

        # Каналы каждого stage
        self.channels = [info["num_chs"] for info in timm_model.feature_info.info]

        # Адаптер: SAR 96ch -> backbone stage0 channels
        # Используем DWConv + pointwise для лучшей адаптации cross-domain features
        stage0_ch = self.channels[0]
        if in_channels != stage0_ch:
            self.stem_adapter = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1,
                          groups=in_channels, bias=False),  # spatial mixing
                nn.Conv2d(in_channels, stage0_ch, 1, bias=False),  # channel proj
                nn.BatchNorm2d(stage0_ch),
                nn.GELU(),
            )
        else:
            self.stem_adapter = nn.Identity()

        # Извлекаем stages (без stem)
        # timm ConvNeXt features_only: stem_0, stages_1..3
        # Нам нужны stages без patchify stem
        self._extract_stages()

    def _extract_stages(self):
        """Извлечь stages из timm модели."""
        # timm ConvNeXt features_only имеет:
        # model.stem (patchify), model.stages[0..3]
        # В features_only mode: feature_info дает indices
        children = list(self.model.children())
        # Обычно: stem, stage0, stage1, stage2, stage3
        # Для ConvNeXt: stem + 4 stages
        self.stages = nn.ModuleList()
        # Пропускаем stem (index 0), берём остальные stage
        for i, child in enumerate(children):
            if i == 0:
                continue  # skip stem
            self.stages.append(child)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: (B, 96, H/4, W/4) — fused features

        Returns:
            features: list of 4 tensors at different scales
                [stage0: (B, 96, H/4, W/4),
                 stage1: (B, 192, H/8, W/8),
                 stage2: (B, 384, H/16, W/16),
                 stage3: (B, 768, H/32, W/32)]
        """
        x = self.stem_adapter(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features
