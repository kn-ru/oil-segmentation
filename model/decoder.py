"""
U-shaped Decoder с SAFB (Skip Attention Fusion Blocks) + Deep Supervision.

На каждом уровне skip-connection:
  1. Upsample deep feature
  2. 1x1 project deep и skip к одинаковым каналам
  3. Gate = sigmoid(conv1x1(cat(deep, skip, abs(deep - skip))))
  4. Fused = DWConv3x3(gate * skip + (1 - gate) * deep)

Deep supervision: auxiliary 1-ch seg outputs на каждом уровне декодера.
Decoder channels: [384, 192, 96]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class SAFBBlock(nn.Module):
    """Skip Attention Fusion Block."""

    def __init__(self, deep_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.proj_deep = nn.Conv2d(deep_channels, out_channels, 1, bias=False)
        self.proj_skip = nn.Conv2d(skip_channels, out_channels, 1, bias=False)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        self.dw_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if deep.shape[-2:] != skip.shape[-2:]:
            deep = F.interpolate(deep, size=skip.shape[-2:],
                                 mode="bilinear", align_corners=False)

        deep_proj = self.proj_deep(deep)
        skip_proj = self.proj_skip(skip)

        diff = torch.abs(deep_proj - skip_proj)
        gate_input = torch.cat([deep_proj, skip_proj, diff], dim=1)
        gate = self.gate_conv(gate_input)

        fused = gate * skip_proj + (1 - gate) * deep_proj
        fused = self.dw_conv(fused)
        return fused


class UDecoder(nn.Module):
    """U-shaped decoder с SAFB блоками и deep supervision.

    backbone_channels: [96, 192, 384, 768]
    decoder_channels:  [384, 192, 96]

    Deep supervision: каждый уровень декодера даёт auxiliary 1-ch logits,
    взвешенные с убывающими коэффициентами [0.25, 0.5, 1.0].
    """

    def __init__(
        self,
        backbone_channels: List[int] = None,
        decoder_channels: List[int] = None,
        bottleneck_channels: int = 768,
        deep_supervision: bool = True,
    ):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [96, 192, 384, 768]
        if decoder_channels is None:
            decoder_channels = [384, 192, 96]

        self.deep_supervision = deep_supervision

        self.safb_blocks = nn.ModuleList()

        in_deep = bottleneck_channels
        for i, dec_ch in enumerate(decoder_channels):
            skip_idx = len(backbone_channels) - 2 - i
            skip_ch = backbone_channels[skip_idx]
            self.safb_blocks.append(SAFBBlock(in_deep, skip_ch, dec_ch))
            in_deep = dec_ch

        self.out_channels = decoder_channels[-1]

        # Deep supervision: auxiliary seg heads на каждом уровне
        if deep_supervision:
            self.aux_heads = nn.ModuleList()
            # Веса: глубже = меньше вес (обучающий сигнал убывает)
            self.ds_weights = [0.25, 0.5, 1.0]  # level0(deepest), level1, level2(final)
            for ch in decoder_channels:
                self.aux_heads.append(nn.Conv2d(ch, 1, 1))

    def forward(self, features: List[torch.Tensor],
                bottleneck_out: torch.Tensor,
                target_size: Optional[tuple] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: list of backbone outputs [stage0, stage1, stage2, stage3]
            bottleneck_out: (B, 768, H/32, W/32)
            target_size: (H, W) для upsampling auxiliary outputs

        Returns:
            dict:
                out: (B, 96, H/4, W/4) — main decoder output
                aux_logits: list of (B, 1, H, W) — deep supervision outputs
                              (upsampled to target_size), весами ds_weights
        """
        x = bottleneck_out
        aux_outputs = []
        intermediate_features = []  # raw decoder features at each level

        n_stages = len(features)
        for i, safb in enumerate(self.safb_blocks):
            skip_idx = n_stages - 2 - i
            skip = features[skip_idx]
            x = safb(x, skip)

            intermediate_features.append(x)  # save raw features before aux head

            if self.deep_supervision:
                aux = self.aux_heads[i](x)
                if target_size is not None:
                    aux = F.interpolate(aux, size=target_size,
                                        mode="bilinear", align_corners=False)
                aux_outputs.append(aux)

        return {
            "out": x,
            "intermediate_features": intermediate_features,  # [(B,384,H,W), (B,192,H,W), (B,96,H,W)]
            "aux_logits": aux_outputs,
            "ds_weights": self.ds_weights if self.deep_supervision else [],
        }
