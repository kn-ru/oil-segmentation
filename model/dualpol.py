"""
DualPolMAFUformerMIL — полная сборка модели.

Pipeline (per tile):
  VV, VH -> DualPolStems -> PolarimetricFusion -> ConvNeXt Backbone
  -> Bottleneck -> UDecoder (deep supervision) -> Multi-scale SegHead
  -> TileEmbedding

Pipeline (per bag):
  16 tile embeddings -> BagClassifier -> 3 class logits
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .stems import DualPolStems
from .fusion import PolarimetricFusion
from .backbone import build_convnext_backbone
from .bottleneck import Bottleneck
from .decoder import UDecoder
from .heads import SegmentationHead, TileEmbedding, BagClassifier


class DualPolMAFUformerMIL(nn.Module):

    def __init__(
        self,
        # Stems
        stem_channels: tuple = (32, 48),
        # Fusion
        fusion_out_channels: int = 96,
        # Backbone
        backbone_name: str = "convnext_small",
        backbone_pretrained: bool = True,
        # Bottleneck
        bottleneck_dim: int = 768,
        bottleneck_heads: int = 8,
        bottleneck_window: int = 7,
        bottleneck_blocks: int = 2,
        # Decoder
        decoder_channels: tuple = (384, 192, 96),
        deep_supervision: bool = True,
        # Embedding
        embed_dim: int = 512,
        mask_stats_dim: int = 4,
        # MIL
        mil_heads: int = 8,
        mil_layers: int = 2,
        mil_dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Dual-pol stems
        self.stems = DualPolStems(stem_channels)

        # 2. Polarimetric fusion
        self.fusion = PolarimetricFusion(stem_channels[-1], fusion_out_channels)

        # 3. ConvNeXt backbone
        self.backbone = build_convnext_backbone(
            backbone_name, fusion_out_channels, backbone_pretrained)
        backbone_channels = self.backbone.channels

        # 4. Bottleneck
        self.bottleneck = Bottleneck(
            dim=backbone_channels[-1],
            num_heads=bottleneck_heads,
            window_size=bottleneck_window,
            num_blocks=bottleneck_blocks,
        )

        # 5. U-Decoder с deep supervision
        self.decoder = UDecoder(
            backbone_channels=backbone_channels,
            decoder_channels=list(decoder_channels),
            bottleneck_channels=backbone_channels[-1],
            deep_supervision=deep_supervision,
        )

        # 6. Multi-scale segmentation head
        # Получает финальные [96] + промежуточные [384, 192] фичи декодера
        # multi_scale_channels: все уровни кроме последнего (он = in_channels)
        ms_channels = list(decoder_channels[:-1]) if len(decoder_channels) > 1 else None
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            upsample_factor=4,
            multi_scale_channels=ms_channels,
        )

        # 7. Tile embedding
        self.tile_embed = TileEmbedding(
            backbone_dim=backbone_channels[-1],
            decoder_dim=decoder_channels[-1],
            mask_stats_dim=mask_stats_dim,
            embed_dim=embed_dim,
        )

        # 8. Bag-level classifier
        self.bag_classifier = BagClassifier(
            embed_dim=embed_dim,
            num_heads=mil_heads,
            num_layers=mil_layers,
            dropout=mil_dropout,
            num_classes=num_classes,
        )

    def forward_tile(self, vv: torch.Tensor, vh: torch.Tensor
                     ) -> Dict[str, torch.Tensor]:
        """Обработка одного тайла (или батча тайлов).

        Args:
            vv: (B, 1, 512, 512)
            vh: (B, 1, 512, 512)

        Returns:
            dict:
                mask_logits: (B, 1, 512, 512)
                tile_embedding: (B, embed_dim)
                aux_logits: list of (B, 1, H, W) — deep supervision
                ds_weights: list of float — weights for aux losses
                decoder_intermediates: list of intermediate decoder features
        """
        # Stems
        f_vv, f_vh = self.stems(vv, vh)

        # Fusion
        f0 = self.fusion(f_vv, f_vh)

        # Backbone
        features = self.backbone(f0)

        # Bottleneck
        bottleneck_out = self.bottleneck(features[-1])

        # Decoder (with deep supervision)
        tile_h, tile_w = vv.shape[-2:]
        dec_result = self.decoder(features, bottleneck_out,
                                  target_size=(tile_h, tile_w))
        decoder_out = dec_result["out"]
        aux_logits = dec_result["aux_logits"]
        ds_weights = dec_result["ds_weights"]
        intermediate_features = dec_result["intermediate_features"]

        # Multi-scale seg head: финальный уровень + промежуточные (если есть ms_fuse)
        # intermediate_features[:-1] — более глубокие уровни [384ch, 192ch]
        # decoder_out = intermediate_features[-1] = 96ch
        ms_feats = intermediate_features[:-1] if len(intermediate_features) > 1 else None
        mask_logits = self.seg_head(decoder_out, multi_scale_features=ms_feats)

        # Tile embedding
        tile_emb = self.tile_embed(features[-1], decoder_out, mask_logits)

        return {
            "mask_logits": mask_logits,
            "tile_embedding": tile_emb,
            "aux_logits": aux_logits,
            "ds_weights": ds_weights,
        }

    def forward(self, vv_tiles: torch.Tensor, vh_tiles: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Полный forward: bag of tiles -> mask logits + class logits.

        Args:
            vv_tiles: (B, N_tiles, 1, 512, 512)
            vh_tiles: (B, N_tiles, 1, 512, 512)

        Returns:
            dict:
                mask_logits:     (B, N_tiles, 1, 512, 512)
                class_logits:    (B, num_classes)
                tile_embeddings: (B, N_tiles, embed_dim)
                aux_logits:      list of (B*N, 1, 512, 512) — deep supervision
                ds_weights:      list of float
        """
        B, N, C, H, W = vv_tiles.shape

        vv_flat = vv_tiles.reshape(B * N, C, H, W)
        vh_flat = vh_tiles.reshape(B * N, C, H, W)

        tile_out = self.forward_tile(vv_flat, vh_flat)

        mask_logits = tile_out["mask_logits"].reshape(B, N, 1, H, W)
        tile_embs = tile_out["tile_embedding"].reshape(B, N, -1)

        # Bag-level classification
        class_logits = self.bag_classifier(tile_embs)

        result = {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "tile_embeddings": tile_embs,
        }

        # Deep supervision outputs (kept flat as B*N for loss computation)
        if self.deep_supervision and tile_out["aux_logits"]:
            result["aux_logits"] = tile_out["aux_logits"]
            result["ds_weights"] = tile_out["ds_weights"]

        return result

    @classmethod
    def from_config(cls, cfg) -> "DualPolMAFUformerMIL":
        m = cfg.model
        return cls(
            stem_channels=tuple(m.stem_channels),
            fusion_out_channels=m.fusion_out_channels,
            backbone_name=m.backbone_name,
            backbone_pretrained=m.backbone_pretrained,
            bottleneck_dim=m.bottleneck_dim,
            bottleneck_heads=m.bottleneck_heads,
            bottleneck_window=m.bottleneck_window,
            bottleneck_blocks=m.bottleneck_blocks,
            decoder_channels=tuple(m.decoder_channels),
            embed_dim=m.embed_dim,
            mask_stats_dim=m.mask_stats_dim,
            mil_heads=m.mil_heads,
            mil_layers=m.mil_layers,
            mil_dropout=m.mil_dropout,
            num_classes=m.num_classes,
        )
