from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimpleConfig:
    # Data
    data_dir: str = "/media/knru/DataLake/OIL"
    tile_size: int = 512
    num_classes: int = 3
    val_fraction: float = 0.15
    clip_percentiles: List[float] = field(default_factory=lambda: [1.0, 99.0])
    force_oil_crop_prob: float = 0.5
    force_oil_min_ratio: float = 0.01
    infer_stride: int = 384

    # Model
    backbone: str = "convnext_tiny"   # ~28M total
    pretrained: bool = True
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Training
    epochs: int = 100
    warmup_epochs: int = 5
    batch_size: int = 4            # tile-level: 4 × 512×512
    grad_accum_steps: int = 4      # effective batch = 16 tiles
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True
    ema_decay: float = 0.999

    # Loss weights
    seg_weight: float = 2.0        # сегментация важнее
    lovasz_weight: float = 0.6
    dice_weight: float = 0.4
    label_smoothing: float = 0.05
    class_weights: List[float] = field(default_factory=lambda: [0.714, 1.25, 1.25])

    # Sampler
    lookalike_oversample: float = 1.5

    # Paths
    save_dir: str = "/home/knru/EcoAir/oil_detection/checkpoints_simple"
    log_dir: str = "/home/knru/EcoAir/oil_detection/runs_simple"
    log_every: int = 20
    val_every: int = 1
    save_every: int = 10

    # Reproducibility
    seed: int = 42

    # Dataloader
    num_workers: int = 4
    pin_memory: bool = True
