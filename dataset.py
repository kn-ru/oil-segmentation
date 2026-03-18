"""
Dataset и тайлинг для Sentinel-1 SAR Oil Spill.

Каждый сэмпл — полное изображение 2048x2048x2, разбитое на bag из 16 тайлов 512x512.
"""

import os
import json
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ──────────────────────────── Нормализация ─────────────────────────────

class SARNormalizer:
    """Клиппинг по перцентилям + z-score нормализация для VV и VH."""

    def __init__(self):
        self.vv_low = None
        self.vv_high = None
        self.vh_low = None
        self.vh_high = None
        self.vv_mean = None
        self.vv_std = None
        self.vh_mean = None
        self.vh_std = None

    def fit(self, image_paths: List[str], percentiles=(1.0, 99.0),
            max_samples: int = 200, seed: int = 42):
        """Вычислить статистику на подмножестве train-данных."""
        rng = np.random.RandomState(seed)
        paths = rng.choice(image_paths, size=min(max_samples, len(image_paths)),
                           replace=False)

        vv_vals, vh_vals = [], []
        for p in paths:
            with rasterio.open(p) as src:
                vv = src.read(1).ravel()
                vh = src.read(2).ravel()
            # Сэмплируем пиксели чтобы не взорвать память
            idx = rng.choice(len(vv), size=min(50000, len(vv)), replace=False)
            vv_vals.append(vv[idx])
            vh_vals.append(vh[idx])

        vv_all = np.concatenate(vv_vals)
        vh_all = np.concatenate(vh_vals)

        self.vv_low, self.vv_high = np.percentile(vv_all, percentiles)
        self.vh_low, self.vh_high = np.percentile(vh_all, percentiles)

        vv_clipped = np.clip(vv_all, self.vv_low, self.vv_high)
        vh_clipped = np.clip(vh_all, self.vh_low, self.vh_high)

        self.vv_mean = float(np.mean(vv_clipped))
        self.vv_std = float(np.std(vv_clipped))
        self.vh_mean = float(np.mean(vh_clipped))
        self.vh_std = float(np.std(vh_clipped))

    def transform(self, vv: np.ndarray, vh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vv = np.clip(vv, self.vv_low, self.vv_high)
        vh = np.clip(vh, self.vh_low, self.vh_high)
        vv = (vv - self.vv_mean) / (self.vv_std + 1e-7)
        vh = (vh - self.vh_mean) / (self.vh_std + 1e-7)
        return vv.astype(np.float32), vh.astype(np.float32)

    def save(self, path: str):
        d = {k: getattr(self, k) for k in
             ["vv_low", "vv_high", "vh_low", "vh_high",
              "vv_mean", "vv_std", "vh_mean", "vh_std"]}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            setattr(self, k, v)
        return self


# ──────────────────────────── Тайлинг ──────────────────────────────────

def image_to_tiles(img: np.ndarray, tile_size: int = 512) -> np.ndarray:
    """Разбить изображение HxWx? на сетку тайлов.

    Args:
        img: (H, W) или (H, W, C)
        tile_size: размер тайла

    Returns:
        (N_tiles, tile_size, tile_size, ...) где N_tiles = (H/tile_size) * (W/tile_size)
    """
    if img.ndim == 2:
        H, W = img.shape
        nh, nw = H // tile_size, W // tile_size
        # Reshape: (nh, tile_size, nw, tile_size) -> (nh*nw, tile_size, tile_size)
        tiles = img[:nh * tile_size, :nw * tile_size]
        tiles = tiles.reshape(nh, tile_size, nw, tile_size)
        tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)
    else:
        H, W, C = img.shape
        nh, nw = H // tile_size, W // tile_size
        tiles = img[:nh * tile_size, :nw * tile_size]
        tiles = tiles.reshape(nh, tile_size, nw, tile_size, C)
        tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, C)
    return tiles


def tiles_to_image(tiles: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Собрать тайлы обратно в изображение."""
    tile_size = tiles.shape[1]
    if tiles.ndim == 3:
        # (N, H, W)
        img = tiles.reshape(grid_h, grid_w, tile_size, tile_size)
        img = img.transpose(0, 2, 1, 3).reshape(grid_h * tile_size, grid_w * tile_size)
    else:
        C = tiles.shape[-1]
        img = tiles.reshape(grid_h, grid_w, tile_size, tile_size, C)
        img = img.transpose(0, 2, 1, 3, 4).reshape(
            grid_h * tile_size, grid_w * tile_size, C)
    return img


# ──────────────────────────── Dataset ──────────────────────────────────

CLASS_DIRS = {
    "oil":       {"images": "Oil",       "masks": "Mask_oil"},
    "lookalike": {"images": "Lookalike", "masks": "Mask_lookalike"},
    "no_oil":    {"images": "No_oil",    "masks": "Mask_no_oil"},
}

CLASS_TO_IDX = {"oil": 0, "lookalike": 1, "no_oil": 2}


def build_file_list(data_dir: str, classes: List[str] = None
                    ) -> List[Dict[str, str]]:
    """Построить список {image_path, mask_path, class_name, class_idx}."""
    if classes is None:
        classes = list(CLASS_DIRS.keys())

    entries = []
    for cls in classes:
        dirs = CLASS_DIRS[cls]
        img_dir = os.path.join(data_dir, dirs["images"])
        mask_dir = os.path.join(data_dir, dirs["masks"])

        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue

        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        common = sorted(img_files & mask_files)

        for fname in common:
            entries.append({
                "image_path": os.path.join(img_dir, fname),
                "mask_path": os.path.join(mask_dir, fname),
                "class_name": cls,
                "class_idx": CLASS_TO_IDX[cls],
                "filename": fname,
            })

    return entries


# ──────────────────────────── SAR Augmentations ────────────────────────

class SARAugmenter:
    """SAR-специфичная аугментация для Sentinel-1 снимков в дБ.

    Все преобразования работают ПОСЛЕ нормализации (z-score).
    Применяется к полному изображению 2048×2048 перед тайлингом.

    Пайплайн:
      1. Геометрия:  rot90, flip H/V                    (всегда вместе с маской)
      2. Speckle:    мультипликативный шум (SAR-специфика)
      3. Radiometry: случайный сдвиг/масштаб dB          (только к изображению)
      4. Noise:      аддитивный гауссов шум
      5. Copy-paste: вставка малых нефтяных патчей (только для oil-класса)
    """

    def __init__(
        self,
        speckle_prob: float = 0.5,    # P(speckle noise)
        speckle_looks: int = 4,        # L-looks модель SAR (меньше = больше шума)
        radio_prob: float = 0.5,       # P(radiometric shift/scale)
        radio_shift: float = 0.3,      # ±σ сдвиг среднего (в нормализованных единицах)
        radio_scale: float = 0.15,     # ±σ масштабирование
        gauss_prob: float = 0.3,       # P(additive Gaussian noise)
        gauss_std: float = 0.1,        # σ гауссова шума
        copypaste_prob: float = 0.5,   # P(copy-paste мелких патчей) — только oil
        copypaste_patches: int = 3,    # сколько патчей вставлять
        copypaste_size: Tuple[int, int] = (64, 256),  # мин/макс размер патча
    ):
        self.speckle_prob = speckle_prob
        self.speckle_looks = speckle_looks
        self.radio_prob = radio_prob
        self.radio_shift = radio_shift
        self.radio_scale = radio_scale
        self.gauss_prob = gauss_prob
        self.gauss_std = gauss_std
        self.copypaste_prob = copypaste_prob
        self.copypaste_patches = copypaste_patches
        self.copypaste_size = copypaste_size

    def __call__(self, vv: np.ndarray, vh: np.ndarray, mask: np.ndarray,
                 is_oil: bool = False,
                 oil_vv_bank: Optional[List[np.ndarray]] = None,
                 oil_vh_bank: Optional[List[np.ndarray]] = None,
                 oil_mask_bank: Optional[List[np.ndarray]] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. Геометрия (вместе с маской)
        vv, vh, mask = self._geometric(vv, vh, mask)

        # 2. Speckle noise (только изображение)
        if np.random.random() < self.speckle_prob:
            vv, vh = self._speckle(vv, vh)

        # 3. Radiometric shift/scale (только изображение)
        if np.random.random() < self.radio_prob:
            vv, vh = self._radiometric(vv, vh)

        # 4. Additive Gaussian noise
        if np.random.random() < self.gauss_prob:
            vv, vh = self._gaussian_noise(vv, vh)

        # 5. Copy-paste нефтяных патчей
        if (is_oil
                and oil_vv_bank is not None
                and np.random.random() < self.copypaste_prob):
            vv, vh, mask = self._copy_paste(vv, vh, mask,
                                             oil_vv_bank, oil_vh_bank, oil_mask_bank)

        return vv, vh, mask

    # ── Геометрия ──────────────────────────────────────────────────────

    def _geometric(self, vv, vh, mask):
        k = np.random.randint(4)
        if k > 0:
            vv   = np.rot90(vv,   k).copy()
            vh   = np.rot90(vh,   k).copy()
            mask = np.rot90(mask, k).copy()
        if np.random.random() > 0.5:
            vv   = np.fliplr(vv).copy()
            vh   = np.fliplr(vh).copy()
            mask = np.fliplr(mask).copy()
        if np.random.random() > 0.5:
            vv   = np.flipud(vv).copy()
            vh   = np.flipud(vh).copy()
            mask = np.flipud(mask).copy()
        return vv, vh, mask

    # ── SAR Speckle Noise ──────────────────────────────────────────────
    #
    # Данные уже z-score нормализованы, поэтому мультипликативный шум
    # некорректен (значения могут быть отрицательными).
    # Вместо этого моделируем speckle как аддитивный шум с амплитудой,
    # зависящей от Gamma(L, 1/L) — это сохраняет дисперсию speckle-модели,
    # но корректно работает в нормализованном пространстве.
    # Эффект: повышенная дисперсия, имитирующая разные числа looks.

    def _speckle(self, vv, vh):
        L = self.speckle_looks
        # g ~ Gamma(L, 1/L), E[g]=1, Var[g]=1/L
        # Используем (g - 1) как аддитивный шум — нулевое среднее, дисперсия 1/L
        g_vv = np.random.gamma(L, 1.0 / L, vv.shape).astype(np.float32) - 1.0
        g_vh = np.random.gamma(L, 1.0 / L, vh.shape).astype(np.float32) - 1.0
        return vv + g_vv, vh + g_vh

    # ── Radiometric Shift + Scale ──────────────────────────────────────
    #
    # Имитирует различия в калибровке, угле падения, сезонных изменениях.
    # В нормализованном пространстве: сдвиг среднего и масштабирование.
    # Разные значения для VV и VH — у них разная физика рассеяния.

    def _radiometric(self, vv, vh):
        # Отдельные сдвиг/масштаб для каждого канала
        for arr in (vv, vh):
            shift = np.random.normal(0, self.radio_shift)
            scale = 1.0 + np.random.normal(0, self.radio_scale)
            scale = np.clip(scale, 0.7, 1.3)  # не слишком экстремально
            arr += shift
            arr *= scale
        return vv, vh

    # ── Gaussian Noise ─────────────────────────────────────────────────

    def _gaussian_noise(self, vv, vh):
        vv = vv + np.random.normal(0, self.gauss_std, vv.shape).astype(np.float32)
        vh = vh + np.random.normal(0, self.gauss_std, vh.shape).astype(np.float32)
        return vv, vh

    # ── Copy-Paste нефтяных патчей ─────────────────────────────────────
    #
    # Для снимков с малым количеством нефти (<1%) вставляем
    # случайные патчи нефти из других oil-снимков.
    # Патч вставляется только в фоновые (не-нефтяные) области,
    # чтобы не перекрывать существующую нефть.

    def _copy_paste(self, vv, vh, mask,
                    bank_vv, bank_vh, bank_mask):
        H, W = vv.shape
        min_sz, max_sz = self.copypaste_size

        for _ in range(self.copypaste_patches):
            # Выбрать случайный источник из банка
            src_idx = np.random.randint(len(bank_vv))
            src_vv   = bank_vv[src_idx]
            src_vh   = bank_vh[src_idx]
            src_mask = bank_mask[src_idx]

            # Найти патч с нефтью в источнике
            patch_found = False
            src_h, src_w = src_vv.shape
            actual_max_sz = min(max_sz, src_h - 1, src_w - 1, H - 1, W - 1)
            if actual_max_sz < min_sz:
                continue
            for _ in range(20):
                sz = np.random.randint(min_sz, actual_max_sz + 1)
                sy = np.random.randint(0, src_h - sz)
                sx = np.random.randint(0, src_w - sz)
                patch_m = src_mask[sy:sy + sz, sx:sx + sz]
                if patch_m.mean() >= 0.05:  # патч содержит ≥5% нефти
                    patch_found = True
                    break

            if not patch_found:
                continue

            # Целевое место — предпочтительно фоновая область
            for _ in range(10):
                ty = np.random.randint(0, H - sz)
                tx = np.random.randint(0, W - sz)
                target_existing = mask[ty:ty + sz, tx:tx + sz].mean()
                if target_existing < 0.1:  # не перекрываем существующую нефть
                    break

            p_vv   = src_vv[sy:sy + sz, sx:sx + sz]
            p_vh   = src_vh[sy:sy + sz, sx:sx + sz]
            p_mask = patch_m

            # Плавные края через маску Гаусса (убирает резкие границы патча)
            blend = _gaussian_blend_mask(sz)

            vv[ty:ty + sz, tx:tx + sz]   = (blend * p_vv +
                                             (1 - blend) * vv[ty:ty + sz, tx:tx + sz])
            vh[ty:ty + sz, tx:tx + sz]   = (blend * p_vh +
                                             (1 - blend) * vh[ty:ty + sz, tx:tx + sz])
            # Маску просто OR — нефть есть нефть
            mask[ty:ty + sz, tx:tx + sz] = np.maximum(
                mask[ty:ty + sz, tx:tx + sz], p_mask)

        return vv, vh, mask


def _gaussian_blend_mask(size: int, sigma_ratio: float = 0.35) -> np.ndarray:
    """2D Гауссова маска для плавного blending копи-пасте патча."""
    sigma = size * sigma_ratio
    center = size / 2.0
    y = np.arange(size) - center
    x = np.arange(size) - center
    xx, yy = np.meshgrid(x, y)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g.astype(np.float32)


# ──────────────────────────── Dataset ──────────────────────────────────

class OilSpillBagDataset(Dataset):
    """
    Bag-level dataset: каждый сэмпл — полное изображение, разбитое на тайлы.

    Returns:
        vv_tiles:  (16, 1, 512, 512) float32
        vh_tiles:  (16, 1, 512, 512) float32
        masks:     (16, 512, 512) float32
        label:     int (0=oil, 1=lookalike, 2=no_oil)
    """

    def __init__(
        self,
        entries: List[Dict],
        normalizer: SARNormalizer,
        tile_size: int = 512,
        force_oil_crop_prob: float = 0.0,
        force_oil_min_ratio: float = 0.01,
        augment: bool = False,
        augmenter: Optional["SARAugmenter"] = None,
        oil_patch_bank: Optional[Dict] = None,
    ):
        self.entries = entries
        self.normalizer = normalizer
        self.tile_size = tile_size
        self.force_oil_crop_prob = force_oil_crop_prob
        self.force_oil_min_ratio = force_oil_min_ratio
        self.augment = augment
        self.augmenter = augmenter or (SARAugmenter() if augment else None)
        # Банк нефтяных патчей для copy-paste: {vv: [...], vh: [...], mask: [...]}
        self.oil_patch_bank = oil_patch_bank

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Загрузка
        with rasterio.open(entry["image_path"]) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)
        with rasterio.open(entry["mask_path"]) as src:
            mask = src.read(1).astype(np.float32)

        # Нормализация
        vv, vh = self.normalizer.transform(vv, vh)

        # Аугментации
        if self.augment and self.augmenter is not None:
            is_oil = entry["class_name"] == "oil"
            bank = self.oil_patch_bank
            vv, vh, mask = self.augmenter(
                vv, vh, mask,
                is_oil=is_oil,
                oil_vv_bank=bank["vv"] if (bank and is_oil) else None,
                oil_vh_bank=bank["vh"] if (bank and is_oil) else None,
                oil_mask_bank=bank["mask"] if (bank and is_oil) else None,
            )

        # Тайлинг
        vv_tiles   = image_to_tiles(vv,   self.tile_size)
        vh_tiles   = image_to_tiles(vh,   self.tile_size)
        mask_tiles = image_to_tiles(mask, self.tile_size)

        # Форсированный кроп с нефтью для oil-класса
        if (entry["class_name"] == "oil"
                and self.force_oil_crop_prob > 0
                and np.random.random() < self.force_oil_crop_prob):
            vv_tiles, vh_tiles, mask_tiles = self._force_oil_tiles(
                vv, vh, mask, vv_tiles, vh_tiles, mask_tiles)

        # (N, H, W) -> (N, 1, H, W)
        vv_tiles = vv_tiles[:, np.newaxis, :, :]
        vh_tiles = vh_tiles[:, np.newaxis, :, :]

        return {
            "vv":        torch.from_numpy(vv_tiles),
            "vh":        torch.from_numpy(vh_tiles),
            "mask":      torch.from_numpy(mask_tiles),
            "label":     entry["class_idx"],
            "filename":  entry["filename"],
            "class_name": entry["class_name"],
        }

    def _force_oil_tiles(self, vv, vh, mask, vv_tiles, vh_tiles, mask_tiles):
        """Заменить случайный тайл на кроп с >min_ratio% нефти."""
        H, W = vv.shape
        ts = self.tile_size

        # Найти тайлы с малым количеством нефти для замены
        oil_ratios = mask_tiles.reshape(mask_tiles.shape[0], -1).mean(axis=1)
        low_oil_idx = np.where(oil_ratios < self.force_oil_min_ratio)[0]

        if len(low_oil_idx) == 0:
            return vv_tiles, vh_tiles, mask_tiles

        # Найти случайный кроп с достаточным количеством нефти
        for _ in range(50):
            y = np.random.randint(0, H - ts)
            x = np.random.randint(0, W - ts)
            crop_mask = mask[y:y + ts, x:x + ts]
            ratio = crop_mask.mean()
            if ratio >= self.force_oil_min_ratio:
                replace_idx = np.random.choice(low_oil_idx)
                vv_tiles[replace_idx] = vv[y:y + ts, x:x + ts]
                vh_tiles[replace_idx] = vh[y:y + ts, x:x + ts]
                mask_tiles[replace_idx] = crop_mask
                break

        return vv_tiles, vh_tiles, mask_tiles


# ──────────────────── Balanced Sampler ─────────────────────────────────

class ClassBalancedSampler(Sampler):
    """Сэмплер с балансировкой классов и оверсэмплингом lookalike."""

    def __init__(self, entries: List[Dict], lookalike_oversample: float = 2.0,
                 seed: int = 42):
        self.entries = entries
        self.seed = seed

        # Группировка по классам
        self.class_indices = {}
        for i, e in enumerate(entries):
            cls = e["class_name"]
            if cls not in self.class_indices:
                self.class_indices[cls] = []
            self.class_indices[cls].append(i)

        # Целевое количество сэмплов на класс
        max_count = max(len(v) for v in self.class_indices.values())
        self.samples_per_class = {}
        for cls, indices in self.class_indices.items():
            if cls == "lookalike":
                self.samples_per_class[cls] = int(max_count * lookalike_oversample)
            else:
                self.samples_per_class[cls] = max_count

        self.total = sum(self.samples_per_class.values())

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        indices = []
        for cls, target_n in self.samples_per_class.items():
            cls_idx = self.class_indices[cls]
            sampled = rng.choice(cls_idx, size=target_n, replace=True)
            indices.extend(sampled.tolist())
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.total

    def set_epoch(self, epoch: int):
        self.seed = self.seed + epoch


# ──────────────────── Oil Patch Bank (для copy-paste) ──────────────────

def build_oil_patch_bank(
    oil_entries: List[Dict],
    normalizer: SARNormalizer,
    max_images: int = 100,
    patch_size: int = 256,
    patches_per_image: int = 5,
    min_oil_ratio: float = 0.05,
    seed: int = 42,
) -> Optional[Dict]:
    """Собрать банк нефтяных патчей из oil-изображений для copy-paste аугментации.

    Загружает случайную выборку oil-снимков, вырезает патчи с достаточным
    количеством нефти и сохраняет в памяти.

    Returns:
        dict с ключами 'vv', 'vh', 'mask' — списки numpy arrays.
        None если нет oil-данных.
    """
    import warnings
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    rng = np.random.RandomState(seed)
    chosen = rng.choice(oil_entries, size=min(max_images, len(oil_entries)),
                        replace=False)

    bank_vv, bank_vh, bank_mask = [], [], []

    for entry in chosen:
        try:
            with rasterio.open(entry["image_path"]) as src:
                vv = src.read(1).astype(np.float32)
                vh = src.read(2).astype(np.float32)
            with rasterio.open(entry["mask_path"]) as src:
                mask = src.read(1).astype(np.float32)
        except Exception:
            continue

        vv, vh = normalizer.transform(vv, vh)
        H, W = vv.shape

        n_found = 0
        for _ in range(patches_per_image * 20):
            if n_found >= patches_per_image:
                break
            sz = patch_size
            if H <= sz or W <= sz:
                continue
            y = rng.randint(0, H - sz)
            x = rng.randint(0, W - sz)
            pm = mask[y:y + sz, x:x + sz]
            if pm.mean() >= min_oil_ratio:
                bank_vv.append(vv[y:y + sz, x:x + sz].copy())
                bank_vh.append(vh[y:y + sz, x:x + sz].copy())
                bank_mask.append(pm.copy())
                n_found += 1

    if not bank_vv:
        return None

    return {"vv": bank_vv, "vh": bank_vh, "mask": bank_mask}


# ──────────────────── Train/Val Split ──────────────────────────────────

def stratified_split(entries: List[Dict], val_fraction: float = 0.15,
                     seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Стратифицированное разбиение по классам (whole-image level)."""
    rng = np.random.RandomState(seed)

    by_class = {}
    for e in entries:
        cls = e["class_name"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(e)

    train_entries, val_entries = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val_entries.extend(items[:n_val])
        train_entries.extend(items[n_val:])

    return train_entries, val_entries
