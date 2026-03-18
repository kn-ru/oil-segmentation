"""
Heads:
  1. Segmentation Head: Conv3x3 -> GELU -> Conv1x1 -> 1ch logits (+ multi-scale)
  2. Tile Embedding: GeM(stage4) + GAP(decoder) + mask_stats -> MLP -> 512d
  3. Bag-level MIL Classifier: Transformer encoder + CLS token -> MLP -> 3 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────── Segmentation Head ────────────────────────────────

class SegmentationHead(nn.Module):
    """Multi-scale segmentation head.

    Принимает фичи с нескольких уровней декодера, объединяет через
    конкатенацию + свёртку, затем Conv3x3 -> GELU -> Conv1x1 -> 1ch.
    """

    def __init__(self, in_channels: int = 96, upsample_factor: int = 4,
                 multi_scale_channels: list = None):
        super().__init__()
        self.upsample_factor = upsample_factor

        if multi_scale_channels:
            # Multi-scale: fuse final (in_channels) + intermediate levels
            total_ch = in_channels + sum(multi_scale_channels)
            self.ms_fuse = nn.Sequential(
                nn.Conv2d(total_ch, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
            )
        else:
            self.ms_fuse = None

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor,
                multi_scale_features: list = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — final decoder output
            multi_scale_features: optional list of features from intermediate levels

        Returns:
            (B, 1, H*up, W*up) — mask logits
        """
        if self.ms_fuse is not None and multi_scale_features:
            target_size = x.shape[-2:]
            aligned = [x]
            for feat in multi_scale_features:
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size,
                                         mode="bilinear", align_corners=False)
                aligned.append(feat)
            x = self.ms_fuse(torch.cat(aligned, dim=1))

        x = self.head(x)
        if self.upsample_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsample_factor,
                              mode="bilinear", align_corners=False)
        return x


# ──────────────────── Generalized Mean Pooling ─────────────────────────

class GeM(nn.Module):
    """Generalized Mean Pooling (fp16-safe)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp p to safe range to prevent pow() overflow in fp16
        p = self.p.clamp(min=1.0, max=6.0)
        # Compute in fp32 to avoid fp16 overflow
        x_fp32 = x.float().clamp(min=self.eps)
        pooled = F.adaptive_avg_pool2d(x_fp32.pow(p), 1).pow(1.0 / p)
        return pooled.squeeze(-1).squeeze(-1).to(x.dtype)


# ──────────────────── Tile Embedding ───────────────────────────────────

class TileEmbedding(nn.Module):
    """
    z1 = GeM(backbone_stage4)
    z2 = GAP(decoder_last)
    mstats = [mean, max, std, pos_ratio]
    z = MLP(cat(z1, z2, mstats)) -> embed_dim
    """

    def __init__(self, backbone_dim: int = 768, decoder_dim: int = 96,
                 mask_stats_dim: int = 4, embed_dim: int = 512):
        super().__init__()
        self.gem = GeM()
        self.gap = nn.AdaptiveAvgPool2d(1)

        total_dim = backbone_dim + decoder_dim + mask_stats_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, backbone_stage4: torch.Tensor,
                decoder_out: torch.Tensor,
                mask_logits: torch.Tensor) -> torch.Tensor:
        z1 = self.gem(backbone_stage4)
        z2 = self.gap(decoder_out).flatten(1)

        mask_probs = torch.sigmoid(mask_logits.squeeze(1))
        mask_mean = mask_probs.mean(dim=(1, 2))
        mask_max = mask_probs.amax(dim=(1, 2))
        mask_std = mask_probs.std(dim=(1, 2))
        pos_ratio = (mask_probs > 0.5).float().mean(dim=(1, 2))

        mstats = torch.stack([mask_mean, mask_max, mask_std, pos_ratio], dim=1)

        z = torch.cat([z1, z2, mstats], dim=1)
        return self.mlp(z)


# ──────────────────── MIL Bag Classifier ───────────────────────────────

class BagClassifier(nn.Module):
    """
    2-layer transformer encoder over tile embeddings + CLS token.
    dim=512, 8 heads, dropout 0.1.
    MLP head: 512 -> 256 -> num_classes.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 17, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        B, N, D = tile_embeddings.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, tile_embeddings], dim=1)

        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :x.shape[1]]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=x.shape[1],
                mode="linear", align_corners=False
            ).transpose(1, 2)
            x = x + pos

        x = self.transformer(x)
        cls_out = x[:, 0]
        return self.head(cls_out)
