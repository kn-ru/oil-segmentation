"""
Анализ полного датасета Sentinel-1 SAR Oil Spill (3 класса + тест).

Классы:
  - oil:       Oil/ + Mask_oil/        (1200 пар)
  - lookalike: Lookalike/ + Mask_lookalike/ (685 пар)
  - no_oil:    No_oil/ + Mask_no_oil/      (685 пар)
  - test:      из Part III              (тест)

Выводит:
1. Посэмпловую статистику по каждому классу (CSV)
2. Сводную статистику в консоль
3. Графики распределений
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from pathlib import Path

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ──────────────────────────── Конфигурация ────────────────────────────

DATA_DIR = "/media/knru/DataLake/OIL"
OUT_DIR = "/home/knru/EcoAir/oil_detection/figures"
CSV_DIR = "/home/knru/EcoAir/oil_detection/analysis"

# Маппинг классов → директории
CLASS_MAP = {
    "oil": {
        "images": os.path.join(DATA_DIR, "Oil"),
        "masks": os.path.join(DATA_DIR, "Mask_oil"),
    },
    "lookalike": {
        "images": os.path.join(DATA_DIR, "Lookalike"),
        "masks": os.path.join(DATA_DIR, "Mask_lookalike"),
    },
    "no_oil": {
        "images": os.path.join(DATA_DIR, "No_oil"),
        "masks": os.path.join(DATA_DIR, "Mask_no_oil"),
    },
}

# Part III — тестовые данные (структура определится после распаковки)
TEST_DIR = os.path.join(DATA_DIR, "Refined Deep‑SAR Oil Spill (SOS) dataset")


# ──────────────────────────── Утилиты ─────────────────────────────────

def load_image(img_path):
    with rasterio.open(img_path) as src:
        vv = src.read(1)
        vh = src.read(2)
    return vv, vh


def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask


def get_common_files(img_dir, mask_dir):
    """Файлы, присутствующие и в img_dir, и в mask_dir."""
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        return []
    masks = set(os.listdir(mask_dir))
    images = set(os.listdir(img_dir))
    common = sorted(masks & images)
    return common


# ──────────── 1. Посэмпловая статистика ───────────────────────────────

def compute_per_sample_stats(files, img_dir, mask_dir, class_name, compute_sar=True):
    """Вычислить статистику по каждому файлу."""
    records = []

    for i, fname in enumerate(files):
        mask = load_mask(os.path.join(mask_dir, fname))

        oil_pixels = int(mask.sum())
        total_pixels = mask.size
        oil_ratio = oil_pixels / total_pixels

        # связные компоненты
        labeled, n_components = ndimage.label(mask)

        component_sizes = []
        if n_components > 0:
            component_sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
            component_sizes = [int(s) for s in component_sizes]

        record = {
            "filename": fname,
            "class": class_name,
            "oil_pixels": oil_pixels,
            "total_pixels": total_pixels,
            "oil_ratio": oil_ratio,
            "n_components": n_components,
            "largest_component": max(component_sizes) if component_sizes else 0,
            "smallest_component": min(component_sizes) if component_sizes else 0,
            "mean_component_size": np.mean(component_sizes) if component_sizes else 0,
        }

        if compute_sar:
            img_path = os.path.join(img_dir, fname)
            if os.path.exists(img_path):
                vv, vh = load_image(img_path)
                oil_mask = mask == 1
                water_mask = mask == 0

                record["vv_mean_all"] = float(np.nanmean(vv))
                record["vv_std_all"] = float(np.nanstd(vv))
                record["vh_mean_all"] = float(np.nanmean(vh))
                record["vh_std_all"] = float(np.nanstd(vh))

                if oil_mask.any():
                    record["vv_mean_oil"] = float(np.nanmean(vv[oil_mask]))
                    record["vv_std_oil"] = float(np.nanstd(vv[oil_mask]))
                    record["vh_mean_oil"] = float(np.nanmean(vh[oil_mask]))
                    record["vh_std_oil"] = float(np.nanstd(vh[oil_mask]))
                    record["vv_vh_diff_oil"] = record["vv_mean_oil"] - record["vh_mean_oil"]

                if water_mask.any():
                    record["vv_mean_water"] = float(np.nanmean(vv[water_mask]))
                    record["vv_std_water"] = float(np.nanstd(vv[water_mask]))
                    record["vh_mean_water"] = float(np.nanmean(vh[water_mask]))
                    record["vh_std_water"] = float(np.nanstd(vh[water_mask]))
                    record["vv_vh_diff_water"] = record["vv_mean_water"] - record["vh_mean_water"]

                if oil_mask.any() and water_mask.any():
                    record["vv_contrast"] = record["vv_mean_water"] - record["vv_mean_oil"]
                    record["vh_contrast"] = record["vh_mean_water"] - record["vh_mean_oil"]

        records.append(record)

        if (i + 1) % 50 == 0:
            print(f"  [{class_name}] Processed {i + 1}/{len(files)} files...")

    return pd.DataFrame(records)


# ──────────── 2. Текстовый отчёт ──────────────────────────────────────

def print_report(df):
    print("=" * 70)
    print("     SENTINEL-1 SAR OIL SPILL — ПОЛНЫЙ АНАЛИЗ (3 класса)")
    print("=" * 70)

    # Общее
    print(f"\n{'─' * 40}")
    print("1. ОБЩАЯ СТАТИСТИКА")
    print(f"{'─' * 40}")
    for cls in ["oil", "lookalike", "no_oil"]:
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            continue
        total_oil = sub["oil_pixels"].sum()
        total_px = sub["total_pixels"].sum()
        ratio = total_oil / total_px if total_px > 0 else 0
        print(f"  {cls:>12s}: {len(sub):5d} пар | "
              f"маскированных пикселей: {ratio:.4%} | "
              f"ср. доля маски: {sub['oil_ratio'].mean():.4%}")
    print(f"  {'ВСЕГО':>12s}: {len(df):5d} пар")

    # По каждому классу
    for cls in ["oil", "lookalike", "no_oil"]:
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            continue

        print(f"\n{'─' * 40}")
        print(f"2. КЛАСС: {cls.upper()}")
        print(f"{'─' * 40}")
        print(f"  Снимков: {len(sub)}")
        print(f"  Доля маски: mean={sub['oil_ratio'].mean():.4%}, "
              f"median={sub['oil_ratio'].median():.4%}, "
              f"min={sub['oil_ratio'].min():.4%}, max={sub['oil_ratio'].max():.4%}")

        nc = sub["n_components"]
        print(f"  Связн. комп.: mean={nc.mean():.1f}, median={nc.median():.0f}, "
              f"min={nc.min()}, max={nc.max()}")

        if "vv_mean_oil" in sub.columns:
            valid = sub.dropna(subset=["vv_mean_all"])
            if len(valid) > 0:
                print(f"  VV mean(all): {valid['vv_mean_all'].mean():.2f} dB")
                print(f"  VH mean(all): {valid['vh_mean_all'].mean():.2f} dB")

            valid_oil = sub.dropna(subset=["vv_mean_oil"])
            if len(valid_oil) > 0:
                print(f"  VV mean(mask): {valid_oil['vv_mean_oil'].mean():.2f} dB")
                print(f"  VH mean(mask): {valid_oil['vh_mean_oil'].mean():.2f} dB")

    print(f"\n{'=' * 70}\n")


# ──────────── 3. Графики ──────────────────────────────────────────────

def plot_class_distribution(df):
    """Столбчатая диаграмма количества снимков по классам."""
    counts = df["class"].value_counts().reindex(["oil", "lookalike", "no_oil"])
    colors = {"oil": "#d62728", "lookalike": "#ff7f0e", "no_oil": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[colors[c] for c in counts.index],
                  edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha="center", fontsize=13, fontweight="bold")
    ax.set_xlabel("Класс", fontsize=12)
    ax.set_ylabel("Количество снимков", fontsize=12)
    ax.set_title("Распределение снимков по классам", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Нефть", "Двойник (lookalike)", "Без нефти"], fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] 01_class_distribution.png")


def plot_mask_ratio_by_class(df):
    """Распределение доли маски по классам."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    classes = ["oil", "lookalike", "no_oil"]
    titles = ["Нефть (oil)", "Двойник (lookalike)", "Без нефти (no_oil)"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    for ax, cls, title, color in zip(axes, classes, titles, colors):
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            ax.set_title(f"{title}\n(нет данных)")
            continue

        ax.hist(sub["oil_ratio"] * 100, bins=50, color=color, edgecolor="black", alpha=0.8)
        med = sub["oil_ratio"].median() * 100
        ax.axvline(med, color="navy", ls="--", lw=2,
                   label=f"Медиана: {med:.2f}%")
        ax.set_xlabel("Доля маски (%)", fontsize=11)
        ax.set_ylabel("Количество снимков", fontsize=11)
        ax.set_title(f"{title} (N={len(sub)})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_mask_ratio_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] 02_mask_ratio_by_class.png")


def plot_sar_comparison(df):
    """Сравнение SAR-характеристик между классами."""
    valid = df.dropna(subset=["vv_mean_all", "vh_mean_all"])
    if len(valid) == 0:
        print("[!] Нет SAR данных, пропускаем")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    palette = {"oil": "#d62728", "lookalike": "#ff7f0e", "no_oil": "#2ca02c"}

    # VV mean по классам
    for cls in ["oil", "lookalike", "no_oil"]:
        sub = valid[valid["class"] == cls]
        if len(sub) > 0:
            axes[0, 0].hist(sub["vv_mean_all"], bins=40, alpha=0.5,
                            label=cls, color=palette[cls])
    axes[0, 0].set_title("Среднее VV по снимку", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Sigma0 VV (дБ)")
    axes[0, 0].set_ylabel("Количество")
    axes[0, 0].legend()

    # VH mean по классам
    for cls in ["oil", "lookalike", "no_oil"]:
        sub = valid[valid["class"] == cls]
        if len(sub) > 0:
            axes[0, 1].hist(sub["vh_mean_all"], bins=40, alpha=0.5,
                            label=cls, color=palette[cls])
    axes[0, 1].set_title("Среднее VH по снимку", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Sigma0 VH (дБ)")
    axes[0, 1].set_ylabel("Количество")
    axes[0, 1].legend()

    # Scatter VV vs VH
    for cls in ["oil", "lookalike", "no_oil"]:
        sub = valid[valid["class"] == cls]
        if len(sub) > 0:
            axes[1, 0].scatter(sub["vv_mean_all"], sub["vh_mean_all"],
                               alpha=0.3, s=10, label=cls, color=palette[cls])
    axes[1, 0].set_xlabel("VV среднее (дБ)")
    axes[1, 0].set_ylabel("VH среднее (дБ)")
    axes[1, 0].set_title("VV vs VH (все пиксели)", fontsize=12, fontweight="bold")
    axes[1, 0].legend()

    # Boxplot VV по классам
    data_bp = []
    for cls in ["oil", "lookalike", "no_oil"]:
        sub = valid[valid["class"] == cls]
        for _, row in sub.iterrows():
            data_bp.append({"Класс": cls, "Канал": "VV", "Sigma0 (дБ)": row["vv_mean_all"]})
            data_bp.append({"Класс": cls, "Канал": "VH", "Sigma0 (дБ)": row["vh_mean_all"]})
    bp_df = pd.DataFrame(data_bp)
    sns.boxplot(data=bp_df, x="Канал", y="Sigma0 (дБ)", hue="Класс",
                palette=palette, ax=axes[1, 1])
    axes[1, 1].set_title("SAR по каналам и классам", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_sar_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] 03_sar_comparison.png")


def plot_components_by_class(df):
    """Распределение числа связных компонент по классам."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    classes = ["oil", "lookalike", "no_oil"]
    titles = ["Нефть", "Двойник", "Без нефти"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    for ax, cls, title, color in zip(axes, classes, titles, colors):
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            ax.set_title(f"{title}\n(нет данных)")
            continue

        max_comp = min(int(sub["n_components"].max()), 80)
        ax.hist(sub["n_components"], bins=range(0, max_comp + 2),
                color=color, edgecolor="black", alpha=0.8)
        ax.set_xlabel("Количество компонент", fontsize=11)
        ax.set_ylabel("Количество снимков", fontsize=11)
        med = sub["n_components"].median()
        ax.set_title(f"{title}: медиана={med:.0f}, макс={sub['n_components'].max()}",
                     fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_components_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] 04_components_by_class.png")


def plot_oil_vs_lookalike_sar(df):
    """Сравнение SAR для маскированных областей oil vs lookalike."""
    oil = df[(df["class"] == "oil")].dropna(subset=["vv_mean_oil"])
    look = df[(df["class"] == "lookalike")].dropna(subset=["vv_mean_oil"])

    if len(oil) == 0 or len(look) == 0:
        print("[!] Недостаточно данных для oil vs lookalike")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(oil["vv_mean_oil"], bins=40, alpha=0.6, label="Нефть", color="#d62728")
    axes[0].hist(look["vv_mean_oil"], bins=40, alpha=0.6, label="Двойник", color="#ff7f0e")
    axes[0].set_title("VV: маскированные области", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sigma0 VV (дБ)")
    axes[0].set_ylabel("Количество")
    axes[0].legend()

    axes[1].hist(oil["vh_mean_oil"], bins=40, alpha=0.6, label="Нефть", color="#d62728")
    axes[1].hist(look["vh_mean_oil"], bins=40, alpha=0.6, label="Двойник", color="#ff7f0e")
    axes[1].set_title("VH: маскированные области", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sigma0 VH (дБ)")
    axes[1].set_ylabel("Количество")
    axes[1].legend()

    plt.suptitle("SAR-сигнатуры: нефть vs двойник (lookalike)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_oil_vs_lookalike_sar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] 05_oil_vs_lookalike_sar.png")


def plot_contrast_comparison(df):
    """Контраст (фон - маска) по классам."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cls, color, label in [("oil", "#d62728", "Нефть"),
                               ("lookalike", "#ff7f0e", "Двойник")]:
        sub = df[df["class"] == cls].dropna(subset=["vv_contrast"])
        if len(sub) > 0:
            axes[0].hist(sub["vv_contrast"], bins=40, alpha=0.6, label=label, color=color)
            axes[1].hist(sub["vh_contrast"], bins=40, alpha=0.6, label=label, color=color)

    axes[0].set_title("Контраст VV (фон − маска)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Контраст (дБ)")
    axes[0].set_ylabel("Количество")
    axes[0].legend()
    axes[0].axvline(0, color="gray", ls="--", alpha=0.5)

    axes[1].set_title("Контраст VH (фон − маска)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Контраст (дБ)")
    axes[1].set_ylabel("Количество")
    axes[1].legend()
    axes[1].axvline(0, color="gray", ls="--", alpha=0.5)

    plt.suptitle("Контраст между маскированными и фоновыми областями", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_contrast_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] 06_contrast_comparison.png")


def plot_correlation_matrix(df):
    """Корреляционная матрица числовых признаков."""
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "total_pixels"]
    corr = df[cols].corr()

    rename = {
        "oil_pixels": "Пикс. маски",
        "oil_ratio": "Доля маски",
        "n_components": "Число комп.",
        "largest_component": "Макс. комп.",
        "smallest_component": "Мин. комп.",
        "mean_component_size": "Ср. размер комп.",
        "vv_mean_all": "VV сред.",
        "vv_std_all": "VV std",
        "vh_mean_all": "VH сред.",
        "vh_std_all": "VH std",
        "vv_mean_oil": "VV маска",
        "vv_std_oil": "VV std маска",
        "vh_mean_oil": "VH маска",
        "vh_std_oil": "VH std маска",
        "vv_vh_diff_oil": "VV−VH маска",
        "vv_mean_water": "VV фон",
        "vv_std_water": "VV std фон",
        "vh_mean_water": "VH фон",
        "vh_std_water": "VH std фон",
        "vv_vh_diff_water": "VV−VH фон",
        "vv_contrast": "Контраст VV",
        "vh_contrast": "Контраст VH",
    }
    corr = corr.rename(index=rename, columns=rename)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, annot_kws={"size": 7})
    ax.set_title("Матрица корреляций признаков (все классы)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_correlation_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] 07_correlation_matrix.png")


# ──────────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Анализ SAR Oil Spill Dataset (3 класса)")
    parser.add_argument("--no-sar", action="store_true",
                        help="Анализировать только маски (быстрее)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Макс. файлов на класс (для тестов)")
    parser.add_argument("--classes", nargs="+", default=["oil", "lookalike", "no_oil"],
                        help="Какие классы анализировать")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    compute_sar = not args.no_sar
    all_dfs = []

    for cls in args.classes:
        if cls not in CLASS_MAP:
            print(f"[!] Неизвестный класс: {cls}, пропускаем")
            continue

        cfg = CLASS_MAP[cls]
        files = get_common_files(cfg["images"], cfg["masks"])
        if not files:
            print(f"[!] Нет данных для класса '{cls}', пропускаем")
            continue

        if args.max_files:
            files = files[:args.max_files]

        print(f"\n{'=' * 50}")
        print(f"Класс: {cls.upper()} — {len(files)} файлов (SAR: {'да' if compute_sar else 'нет'})")
        print(f"{'=' * 50}")

        df_cls = compute_per_sample_stats(files, cfg["images"], cfg["masks"],
                                          cls, compute_sar=compute_sar)
        all_dfs.append(df_cls)

        # Сохранить CSV по классу
        csv_path = os.path.join(CSV_DIR, f"stats_{cls}.csv")
        df_cls.to_csv(csv_path, index=False)
        print(f"  [✓] {csv_path}")

    if not all_dfs:
        print("Нет данных для анализа!")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    # Общий CSV
    csv_all = os.path.join(CSV_DIR, "dataset_stats_all.csv")
    df.to_csv(csv_all, index=False)
    print(f"\n[✓] Общий CSV: {csv_all} ({len(df)} записей)")

    # Отчёт
    print_report(df)

    # Графики
    print("\nГенерация графиков...")
    plot_class_distribution(df)
    plot_mask_ratio_by_class(df)
    plot_components_by_class(df)
    if compute_sar:
        plot_sar_comparison(df)
        plot_oil_vs_lookalike_sar(df)
        plot_contrast_comparison(df)
        plot_correlation_matrix(df)

    print(f"\nГотово! Все выходные данные:")
    print(f"  Графики: {OUT_DIR}/")
    print(f"  CSV:     {CSV_DIR}/")


if __name__ == "__main__":
    main()
