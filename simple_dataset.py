"""
Tile-level dataset для SimpleOilNet.

Каждый сэмпл — один тайл 512×512:
  image: (2, 512, 512) float32  ← VV и VH как 2 канала
  mask:  (512, 512)   float32   ← бинарная маска нефти
  label: int                    ← класс изображения (0=oil,1=lookalike,2=no_oil)
"""

import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from dataset import (
    SARNormalizer, SARAugmenter, build_oil_patch_bank,
    image_to_tiles, build_file_list, stratified_split,
    CLASS_TO_IDX,
)


class TileDataset(Dataset):
    """
    Tile-level dataset: возвращает один тайл 512×512 за раз.
    Аугментации применяются на уровне полного изображения,
    затем нарезаем на тайлы.
    """

    def __init__(
        self,
        entries: List[Dict],
        normalizer: SARNormalizer,
        tile_size: int = 512,
        augment: bool = False,
        augmenter: Optional[SARAugmenter] = None,
        oil_patch_bank: Optional[Dict] = None,
        force_oil_crop_prob: float = 0.0,
        force_oil_min_ratio: float = 0.01,
    ):
        self.entries = entries
        self.normalizer = normalizer
        self.tile_size = tile_size
        self.augment = augment
        self.augmenter = augmenter or (SARAugmenter() if augment else None)
        self.oil_patch_bank = oil_patch_bank
        self.force_oil_crop_prob = force_oil_crop_prob
        self.force_oil_min_ratio = force_oil_min_ratio

        # Построить flat список (entry_idx, tile_idx)
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[int, int]]:
        ts = self.tile_size
        samples = []
        for i, entry in enumerate(self.entries):
            # 2048 / 512 = 4×4 = 16 тайлов
            n_tiles = (2048 // ts) ** 2
            for t in range(n_tiles):
                samples.append((i, t))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry_idx, tile_idx = self.samples[idx]
        entry = self.entries[entry_idx]

        with rasterio.open(entry["image_path"]) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)
        with rasterio.open(entry["mask_path"]) as src:
            mask = src.read(1).astype(np.float32)

        vv, vh = self.normalizer.transform(vv, vh)

        if self.augment and self.augmenter is not None:
            is_oil = entry["class_name"] == "oil"
            bank = self.oil_patch_bank
            vv, vh, mask = self.augmenter(
                vv, vh, mask, is_oil=is_oil,
                oil_vv_bank=bank["vv"] if (bank and is_oil) else None,
                oil_vh_bank=bank["vh"] if (bank and is_oil) else None,
                oil_mask_bank=bank["mask"] if (bank and is_oil) else None,
            )

        # Force oil crop
        if (entry["class_name"] == "oil"
                and self.force_oil_crop_prob > 0
                and np.random.random() < self.force_oil_crop_prob):
            vv, vh, mask, tile_idx = self._force_oil_tile(vv, vh, mask, tile_idx)

        # Нарезка
        ts = self.tile_size
        n_cols = 2048 // ts
        row, col = tile_idx // n_cols, tile_idx % n_cols
        y, x = row * ts, col * ts

        vv_tile  = vv[y:y + ts, x:x + ts]
        vh_tile  = vh[y:y + ts, x:x + ts]
        mask_tile = mask[y:y + ts, x:x + ts]

        # VV + VH → 2 канала (contiguous для DataLoader)
        image = np.ascontiguousarray(np.stack([vv_tile, vh_tile], axis=0))
        mask_tile = np.ascontiguousarray(mask_tile)

        return {
            "image": torch.from_numpy(image),
            "mask":  torch.from_numpy(mask_tile),
            "label": entry["class_idx"],
            "filename": entry["filename"],
        }

    def _force_oil_tile(self, vv, vh, mask, tile_idx):
        """Попробовать найти тайл с нефтью."""
        ts = self.tile_size
        H, W = vv.shape
        for _ in range(50):
            y = np.random.randint(0, H - ts)
            x = np.random.randint(0, W - ts)
            if mask[y:y + ts, x:x + ts].mean() >= self.force_oil_min_ratio:
                # Вернуть случайный grid-aligned тайл ближайший к найденному
                n_cols = W // ts
                row = min(y // ts, H // ts - 1)
                col = min(x // ts, W // ts - 1)
                return vv, vh, mask, row * n_cols + col
        return vv, vh, mask, tile_idx


class ClassBalancedTileSampler(Sampler):
    """Сэмплер с балансировкой на уровне тайлов по классам изображений."""

    def __init__(self, dataset: TileDataset, lookalike_oversample: float = 1.5,
                 seed: int = 42):
        self.dataset = dataset
        self.seed = seed

        # Группировать тайлы по классу изображения
        by_class = {}
        for idx, (entry_idx, _) in enumerate(dataset.samples):
            cls = dataset.entries[entry_idx]["class_name"]
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(idx)

        max_n = max(len(v) for v in by_class.values())
        self.samples_per_class = {}
        for cls, idxs in by_class.items():
            if cls == "lookalike":
                self.samples_per_class[cls] = (idxs, int(max_n * lookalike_oversample))
            else:
                self.samples_per_class[cls] = (idxs, max_n)

        self.total = sum(n for _, n in self.samples_per_class.values())

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        indices = []
        for cls, (idxs, n) in self.samples_per_class.items():
            sampled = rng.choice(idxs, size=n, replace=True)
            indices.extend(sampled.tolist())
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.total

    def set_epoch(self, epoch: int):
        self.seed += epoch
