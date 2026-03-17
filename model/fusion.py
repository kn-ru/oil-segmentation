"""
Polarimetric Fusion Block.

gate = sigmoid(conv1x1(cat(Fvv, Fvh, abs(Fvv - Fvh))))
F0 = conv3x3(cat(gate * Fvv, (1 - gate) * Fvh))
output: 96 channels
"""

import torch
import torch.nn as nn


class PolarimetricFusion(nn.Module):

    def __init__(self, in_channels: int = 48, out_channels: int = 96):
        super().__init__()
        # Gate: 3 * in_channels -> 1 (per-spatial gate)
        # Используем in_channels каналов на gate для rich gating
        self.gate_conv = nn.Conv2d(in_channels * 3, in_channels, 1, bias=True)

        # Fusion: 2 * in_channels -> out_channels
        self.fusion_conv = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, f_vv: torch.Tensor, f_vh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_vv: (B, C, H, W) — features from VV stem
            f_vh: (B, C, H, W) — features from VH stem
        Returns:
            (B, out_channels, H, W)
        """
        diff = torch.abs(f_vv - f_vh)
        gate_input = torch.cat([f_vv, f_vh, diff], dim=1)  # (B, 3C, H, W)
        gate = torch.sigmoid(self.gate_conv(gate_input))     # (B, C, H, W)

        fused_input = torch.cat([gate * f_vv, (1 - gate) * f_vh], dim=1)  # (B, 2C, H, W)
        f0 = self.act(self.norm(self.fusion_conv(fused_input)))
        return f0
