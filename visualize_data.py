"""
Визуализация данных Sentinel-1 SAR Oil Spill Dataset.

Отображает:
1. Сетку примеров: SAR VV, SAR VH, маска, наложение маски на VV
2. Примеры с наибольшей и наименьшей площадью разлива
3. Гистограммы значений VV/VH для областей нефти и воды
"""

import os
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ──────────────────────────── Конфигурация ────────────────────────────

DATA_DIR = "/media/knru/DataLake/OIL"
IMG_DIR = os.path.join(DATA_DIR, "Oil")
MASK_DIR = os.path.join(DATA_DIR, "Mask_oil")
OUT_DIR = "/home/knru/EcoAir/oil_detection/figures"


# ──────────────────────────── Утилиты ─────────────────────────────────

def load_image(img_path):
    """Загрузка SAR-изображения (2 канала: VV, VH) в дБ."""
    with rasterio.open(img_path) as src:
        vv = src.read(1)  # band 1 — VV
        vh = src.read(2)  # band 2 — VH
    return vv, vh


def load_mask(mask_path):
    """Загрузка бинарной маски."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask


def get_paired_files(n=None, sort_by_oil=False):
    """Получить пары (image, mask) файлов. Опционально сортировать по доле нефти."""
    mask_files = sorted(os.listdir(MASK_DIR))
    img_files_set = set(os.listdir(IMG_DIR))

    pairs = []
    for mf in mask_files:
        if mf in img_files_set:
            pairs.append(mf)

    if sort_by_oil:
        oil_ratios = []
        for f in pairs:
            m = load_mask(os.path.join(MASK_DIR, f))
            oil_ratios.append(m.sum() / m.size)
        order = np.argsort(oil_ratios)
        pairs = [pairs[i] for i in order]
        oil_ratios = [oil_ratios[i] for i in order]
        if n:
            return pairs[:n], pairs[-n:], oil_ratios
        return pairs, None, oil_ratios

    if n:
        # выбрать равномерно распределённые индексы
        indices = np.linspace(0, len(pairs) - 1, n, dtype=int)
        pairs = [pairs[i] for i in indices]

    return pairs


def normalize_for_display(band):
    """Нормализация SAR-канала для отображения (клиппинг по перцентилям)."""
    p2, p98 = np.nanpercentile(band, [2, 98])
    clipped = np.clip(band, p2, p98)
    return (clipped - p2) / (p98 - p2 + 1e-10)


# ──────────────────────── 1. Сетка примеров ───────────────────────────

def plot_sample_grid(n_samples=6, seed=42):
    """Сетка: VV | VH | Маска | VV + наложение маски."""
    np.random.seed(seed)
    files = get_paired_files()
    chosen = np.random.choice(files, size=min(n_samples, len(files)), replace=False)

    fig, axes = plt.subplots(len(chosen), 4, figsize=(20, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    titles = ["SAR VV (Sigma0 dB)", "SAR VH (Sigma0 dB)", "Ground Truth Mask", "VV + Oil Overlay"]

    for i, fname in enumerate(chosen):
        vv, vh = load_image(os.path.join(IMG_DIR, fname))
        mask = load_mask(os.path.join(MASK_DIR, fname))

        vv_norm = normalize_for_display(vv)
        vh_norm = normalize_for_display(vh)

        # VV
        axes[i, 0].imshow(vv_norm, cmap="gray")
        axes[i, 0].set_ylabel(fname.replace(".tif", ""), fontsize=11, fontweight="bold")

        # VH
        axes[i, 1].imshow(vh_norm, cmap="gray")

        # Маска
        axes[i, 2].imshow(mask, cmap="RdYlBu_r", vmin=0, vmax=1)

        # Наложение
        axes[i, 3].imshow(vv_norm, cmap="gray")
        overlay = np.ma.masked_where(mask == 0, mask)
        axes[i, 3].imshow(overlay, cmap="autumn", alpha=0.5, vmin=0, vmax=1)

        oil_pct = mask.sum() / mask.size * 100
        axes[i, 3].text(
            50, 100, f"Oil: {oil_pct:.2f}%",
            color="white", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
        )

    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Sentinel-1 SAR Oil Spill — Sample Grid", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_sample_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 01_sample_grid.png saved")


# ──────────── 2. Экстремальные примеры (мин/макс площади) ────────────

def plot_extremes(n=4):
    """Примеры с наименьшей и наибольшей долей нефти."""
    smallest, largest, ratios = get_paired_files(n=n, sort_by_oil=True)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    for col, fname in enumerate(smallest):
        vv, _ = load_image(os.path.join(IMG_DIR, fname))
        mask = load_mask(os.path.join(MASK_DIR, fname))
        vv_norm = normalize_for_display(vv)

        axes[0, col].imshow(vv_norm, cmap="gray")
        overlay = np.ma.masked_where(mask == 0, mask)
        axes[0, col].imshow(overlay, cmap="autumn", alpha=0.5)
        oil_pct = mask.sum() / mask.size * 100
        axes[0, col].set_title(f"{fname}\nOil: {oil_pct:.3f}%", fontsize=10)

    for col, fname in enumerate(largest):
        vv, _ = load_image(os.path.join(IMG_DIR, fname))
        mask = load_mask(os.path.join(MASK_DIR, fname))
        vv_norm = normalize_for_display(vv)

        axes[1, col].imshow(vv_norm, cmap="gray")
        overlay = np.ma.masked_where(mask == 0, mask)
        axes[1, col].imshow(overlay, cmap="autumn", alpha=0.5)
        oil_pct = mask.sum() / mask.size * 100
        axes[1, col].set_title(f"{fname}\nOil: {oil_pct:.2f}%", fontsize=10)

    axes[0, 0].set_ylabel("Min oil area", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Max oil area", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Extreme Cases: Smallest vs Largest Oil Spills", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_extremes.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 02_extremes.png saved")


# ──────────── 3. Гистограммы VV/VH: нефть vs вода ────────────────────

def plot_pixel_histograms(n_samples=50, seed=42):
    """Распределения значений VV и VH отдельно для пикселей нефти и воды."""
    np.random.seed(seed)
    files = get_paired_files()
    chosen = np.random.choice(files, size=min(n_samples, len(files)), replace=False)

    vv_oil, vv_water = [], []
    vh_oil, vh_water = [], []

    for fname in chosen:
        vv, vh = load_image(os.path.join(IMG_DIR, fname))
        mask = load_mask(os.path.join(MASK_DIR, fname))

        oil_mask = mask == 1
        water_mask = mask == 0

        # сэмплируем, чтобы не перегружать память
        n_sample = min(5000, oil_mask.sum(), water_mask.sum())
        if n_sample < 100:
            continue

        oil_idx = np.random.choice(np.where(oil_mask.ravel())[0], n_sample, replace=False)
        water_idx = np.random.choice(np.where(water_mask.ravel())[0], n_sample, replace=False)

        vv_oil.append(vv.ravel()[oil_idx])
        vv_water.append(vv.ravel()[water_idx])
        vh_oil.append(vh.ravel()[oil_idx])
        vh_water.append(vh.ravel()[water_idx])

    vv_oil = np.concatenate(vv_oil)
    vv_water = np.concatenate(vv_water)
    vh_oil = np.concatenate(vh_oil)
    vh_water = np.concatenate(vh_water)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # VV
    axes[0].hist(vv_water, bins=200, alpha=0.6, label="Water", color="steelblue", density=True)
    axes[0].hist(vv_oil, bins=200, alpha=0.6, label="Oil", color="red", density=True)
    axes[0].set_title("VV polarization (Sigma0 dB)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Sigma0 (dB)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=12)

    # VH
    axes[1].hist(vh_water, bins=200, alpha=0.6, label="Water", color="steelblue", density=True)
    axes[1].hist(vh_oil, bins=200, alpha=0.6, label="Oil", color="red", density=True)
    axes[1].set_title("VH polarization (Sigma0 dB)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Sigma0 (dB)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=12)

    plt.suptitle("Pixel Value Distributions: Oil vs Water", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_pixel_histograms.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 03_pixel_histograms.png saved")


# ──────────── 4. Визуализация VV-VH разности ─────────────────────────

def plot_vv_vh_difference(n_samples=4, seed=42):
    """Показать разность VV-VH — часто помогает различать нефть и воду."""
    np.random.seed(seed)
    files = get_paired_files()
    chosen = np.random.choice(files, size=min(n_samples, len(files)), replace=False)

    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    for i, fname in enumerate(chosen):
        vv, vh = load_image(os.path.join(IMG_DIR, fname))
        mask = load_mask(os.path.join(MASK_DIR, fname))

        diff = vv - vh
        diff_norm = normalize_for_display(diff)
        vv_norm = normalize_for_display(vv)

        axes[i, 0].imshow(vv_norm, cmap="gray")
        axes[i, 0].set_ylabel(fname.replace(".tif", ""), fontsize=11, fontweight="bold")

        axes[i, 1].imshow(diff_norm, cmap="viridis")

        axes[i, 2].imshow(diff_norm, cmap="viridis")
        contour_mask = mask.astype(float)
        axes[i, 2].contour(contour_mask, levels=[0.5], colors="red", linewidths=1)

    axes[0, 0].set_title("SAR VV", fontsize=13, fontweight="bold")
    axes[0, 1].set_title("VV − VH (difference)", fontsize=13, fontweight="bold")
    axes[0, 2].set_title("VV − VH + Mask contour", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("VV − VH Difference Map", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_vv_vh_difference.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 04_vv_vh_difference.png saved")


# ──────────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Визуализация SAR Oil Spill Dataset")
    parser.add_argument("--all", action="store_true", help="Построить все графики")
    parser.add_argument("--grid", action="store_true", help="Сетка примеров")
    parser.add_argument("--extremes", action="store_true", help="Экстремальные примеры")
    parser.add_argument("--hist", action="store_true", help="Гистограммы пикселей")
    parser.add_argument("--diff", action="store_true", help="Карта VV-VH разности")
    parser.add_argument("--n-samples", type=int, default=6, help="Число примеров в сетке")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    run_all = args.all or not any([args.grid, args.extremes, args.hist, args.diff])

    if run_all or args.grid:
        plot_sample_grid(n_samples=args.n_samples)
    if run_all or args.extremes:
        plot_extremes()
    if run_all or args.hist:
        plot_pixel_histograms()
    if run_all or args.diff:
        plot_vv_vh_difference()

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
