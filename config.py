"""
Конфигурация DualPolMAFUformerMIL.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    data_dir: str = "/media/knru/DataLake/OIL"
    tile_size: int = 512
    tiles_per_bag: int = 16  # 2048/512 = 4x4
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["oil", "lookalike", "no_oil"])

    # Нормализация: перцентили вычисляются на train split
    clip_percentiles: List[float] = field(default_factory=lambda: [1.0, 99.0])

    # Train/val split
    val_fraction: float = 0.15
    seed: int = 42

    # Форсированный кроп с нефтью (для oil-класса)
    force_oil_crop_prob: float = 0.5
    force_oil_min_ratio: float = 0.01

    # Inference
    infer_stride: int = 384

    # Dataloader
    batch_size: int = 2  # bag-level: 2 images = 32 tiles
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    # Stems
    stem_channels: List[int] = field(default_factory=lambda: [32, 48])

    # Fusion
    fusion_out_channels: int = 96

    # Backbone: ConvNeXt-Small
    backbone_name: str = "convnext_small"  # fallback: convnext_tiny
    backbone_channels: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    backbone_pretrained: bool = True

    # Bottleneck
    bottleneck_dim: int = 768
    bottleneck_heads: int = 8
    bottleneck_window: int = 7
    bottleneck_blocks: int = 2

    # Decoder
    decoder_channels: List[int] = field(default_factory=lambda: [384, 192, 96])

    # Tile embedding
    embed_dim: int = 512
    mask_stats_dim: int = 4  # mean, max, std, positive_area_ratio

    # MIL transformer
    mil_heads: int = 8
    mil_layers: int = 2
    mil_dropout: float = 0.1

    # Classification head
    num_classes: int = 3


@dataclass
class TrainConfig:
    epochs: int = 150
    warmup_epochs: int = 5

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    # Gradient accumulation: effective_batch = batch_size * grad_accum_steps
    grad_accum_steps: int = 4  # effective batch = 2*4 = 8 bags = 128 tiles

    # EMA
    ema_decay: float = 0.999

    # Loss weights
    seg_weight: float = 0.7
    focal_weight: float = 0.5
    dice_weight: float = 0.3
    boundary_weight: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    label_smoothing: float = 0.05
    ohem_ratio: float = 0.7  # top 70% hardest pixels
    ds_weight: float = 0.4   # deep supervision weight

    # Class weights (inverse frequency): oil=1200, lookalike=685, no_oil=685
    # Computed as: N_total / (N_classes * N_class_i)
    # oil: 2570/(3*1200)=0.714, lookalike: 2570/(3*685)=1.250, no_oil: same
    class_weights: List[float] = field(default_factory=lambda: [0.714, 1.250, 1.250])

    # AMP
    use_amp: bool = True

    # Lookalike oversampling factor
    lookalike_oversample: float = 1.5  # reduced from 2.0 (class weights handle balance)

    # Logging
    log_dir: str = "/home/knru/EcoAir/oil_detection/runs"
    save_dir: str = "/home/knru/EcoAir/oil_detection/checkpoints"
    log_every: int = 10
    val_every: int = 1
    save_every: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
