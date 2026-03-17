"""
Dual polarization stems: отдельные неглубокие энкодеры для VV и VH каналов.

Conv3x3 stride2 1->32, norm, GELU
Conv3x3 stride2 32->48, norm, GELU

Входное разрешение 512 -> 128 (stride 4x)
"""

import torch
import torch.nn as nn


class PolarizationStem(nn.Module):
    """Shallow stem для одного поляризационного канала."""

    def __init__(self, in_channels: int = 1, channels: tuple = (32, 48)):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv2d(in_channels, c1, 3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(c1)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(c2)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            (B, 48, H/4, W/4)
        """
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class DualPolStems(nn.Module):
    """Два параллельных стема для VV и VH."""

    def __init__(self, channels: tuple = (32, 48)):
        super().__init__()
        self.stem_vv = PolarizationStem(1, channels)
        self.stem_vh = PolarizationStem(1, channels)

    def forward(self, vv: torch.Tensor, vh: torch.Tensor):
        """
        Args:
            vv: (B, 1, H, W)
            vh: (B, 1, H, W)
        Returns:
            f_vv: (B, 48, H/4, W/4)
            f_vh: (B, 48, H/4, W/4)
        """
        return self.stem_vv(vv), self.stem_vh(vh)
