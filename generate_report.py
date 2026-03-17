"""
Генерация полного Markdown-отчёта по датасету Sentinel-1 SAR Oil Spill.
Все 3 класса: oil, lookalike, no_oil.

Включает:
- Таблицы и статистику
- Графики распределений
- Примеры изображений для каждого класса
- Конвертацию в PDF
"""

import os
import warnings
import numpy as np
import pandas as pd
import rasterio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from weasyprint import HTML
import markdown

matplotlib.rcParams["font.family"] = "DejaVu Sans"
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ──────────────────────────── Конфигурация ────────────────────────────

DATA_DIR = "/media/knru/DataLake/OIL"
CSV_PATH = "/home/knru/EcoAir/oil_detection/analysis/dataset_stats_all.csv"
FIGURES_SRC = "/home/knru/EcoAir/oil_detection/figures"
OUT_DIR = "/home/knru/EcoAir/oil_detection/report"
FIG_DIR = os.path.join(OUT_DIR, "figures")
REPORT_PATH = os.path.join(OUT_DIR, "report.md")
PDF_PATH = os.path.join(OUT_DIR, "report.pdf")

CLASS_MAP = {
    "oil": {
        "images": os.path.join(DATA_DIR, "Oil"),
        "masks": os.path.join(DATA_DIR, "Mask_oil"),
        "name_ru": "Нефть",
        "color": "#d62728",
    },
    "lookalike": {
        "images": os.path.join(DATA_DIR, "Lookalike"),
        "masks": os.path.join(DATA_DIR, "Mask_lookalike"),
        "name_ru": "Двойник (lookalike)",
        "color": "#ff7f0e",
    },
    "no_oil": {
        "images": os.path.join(DATA_DIR, "No_oil"),
        "masks": os.path.join(DATA_DIR, "Mask_no_oil"),
        "name_ru": "Без нефти",
        "color": "#2ca02c",
    },
}


# ──────────────────────── Утилиты ──────────────────────────────────────

def load_image(img_path):
    with rasterio.open(img_path) as src:
        vv = src.read(1)
        vh = src.read(2)
    return vv, vh


def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)


def normalize_for_display(band):
    p2, p98 = np.nanpercentile(band, [2, 98])
    clipped = np.clip(band, p2, p98)
    return (clipped - p2) / (p98 - p2 + 1e-10)


def rel(path):
    return os.path.relpath(path, OUT_DIR)


def get_common_files(img_dir, mask_dir):
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        return []
    masks = set(os.listdir(mask_dir))
    images = set(os.listdir(img_dir))
    return sorted(masks & images)


# ──────────────── Графики для отчёта (рус.) ────────────────────────────

def fig_samples_per_class(df, n=3, seed=42):
    """Примеры изображений для каждого класса: VV | VH | Маска | VV + наложение."""
    np.random.seed(seed)
    all_paths = {}

    for cls, cfg in CLASS_MAP.items():
        img_dir = cfg["images"]
        mask_dir = cfg["masks"]
        files = get_common_files(img_dir, mask_dir)
        if not files:
            continue

        # Для oil — выбираем с разной долей нефти
        if cls == "oil":
            sub = df[df["class"] == cls].sort_values("oil_ratio")
            # малое, среднее, большое пятно
            indices = [len(sub)//10, len(sub)//2, int(len(sub)*0.9)]
            chosen = [sub.iloc[i]["filename"] for i in indices]
        else:
            chosen = list(np.random.choice(files, size=min(n, len(files)), replace=False))

        fig, axes = plt.subplots(len(chosen), 4, figsize=(20, 4.5 * len(chosen)))
        if len(chosen) == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["SAR VV (Sigma0 дБ)", "SAR VH (Sigma0 дБ)",
                       "Маска (ground truth)", "VV + наложение маски"]

        for i, fname in enumerate(chosen):
            vv, vh = load_image(os.path.join(img_dir, fname))
            mask = load_mask(os.path.join(mask_dir, fname))

            vv_norm = normalize_for_display(vv)
            vh_norm = normalize_for_display(vh)

            axes[i, 0].imshow(vv_norm, cmap="gray")
            axes[i, 0].set_ylabel(fname.replace(".tif", ""), fontsize=11, fontweight="bold")

            axes[i, 1].imshow(vh_norm, cmap="gray")

            axes[i, 2].imshow(mask, cmap="RdYlBu_r", vmin=0, vmax=1)

            axes[i, 3].imshow(vv_norm, cmap="gray")
            overlay = np.ma.masked_where(mask == 0, mask)
            axes[i, 3].imshow(overlay, cmap="autumn", alpha=0.5, vmin=0, vmax=1)

            pct = mask.sum() / mask.size * 100
            axes[i, 3].text(
                50, 100, f"Маска: {pct:.2f}%",
                color="white", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
            )

        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=12, fontweight="bold")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        name_ru = cfg["name_ru"]
        plt.suptitle(f"Примеры: {name_ru} (класс {cls})",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"samples_{cls}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        all_paths[cls] = path
        print(f"  [+] Примеры: {cls}")

    return all_paths


def fig_extremes_all_classes(df, n=3):
    """Экстремальные примеры: наибольшая и наименьшая доля маски по каждому классу."""
    all_paths = {}

    for cls, cfg in CLASS_MAP.items():
        img_dir = cfg["images"]
        mask_dir = cfg["masks"]

        sub = df[df["class"] == cls].sort_values("oil_ratio")
        if len(sub) < 2 * n:
            continue

        smallest = list(sub.head(n)["filename"])
        largest = list(sub.tail(n)["filename"])

        fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

        for col, fname in enumerate(smallest):
            vv, _ = load_image(os.path.join(img_dir, fname))
            mask = load_mask(os.path.join(mask_dir, fname))
            vv_norm = normalize_for_display(vv)
            axes[0, col].imshow(vv_norm, cmap="gray")
            overlay = np.ma.masked_where(mask == 0, mask)
            axes[0, col].imshow(overlay, cmap="autumn", alpha=0.5)
            pct = mask.sum() / mask.size * 100
            axes[0, col].set_title(f"{fname}\nМаска: {pct:.3f}%", fontsize=10)

        for col, fname in enumerate(largest):
            vv, _ = load_image(os.path.join(img_dir, fname))
            mask = load_mask(os.path.join(mask_dir, fname))
            vv_norm = normalize_for_display(vv)
            axes[1, col].imshow(vv_norm, cmap="gray")
            overlay = np.ma.masked_where(mask == 0, mask)
            axes[1, col].imshow(overlay, cmap="autumn", alpha=0.5)
            pct = mask.sum() / mask.size * 100
            axes[1, col].set_title(f"{fname}\nМаска: {pct:.2f}%", fontsize=10)

        axes[0, 0].set_ylabel("Мин. площадь", fontsize=13, fontweight="bold")
        axes[1, 0].set_ylabel("Макс. площадь", fontsize=13, fontweight="bold")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        name_ru = cfg["name_ru"]
        plt.suptitle(f"Экстремальные примеры: {name_ru}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"extremes_{cls}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        all_paths[cls] = path
        print(f"  [+] Экстремальные: {cls}")

    return all_paths


def fig_pixel_histograms_3class(df, n_samples=30, seed=42):
    """Гистограммы пикселей VV и VH для маскированных и фоновых областей по классам."""
    np.random.seed(seed)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    classes = ["oil", "lookalike", "no_oil"]
    titles = ["Нефть", "Двойник", "Без нефти"]

    for col_idx, (cls, title) in enumerate(zip(classes, titles)):
        cfg = CLASS_MAP[cls]
        files = get_common_files(cfg["images"], cfg["masks"])
        if not files:
            continue

        chosen = np.random.choice(files, size=min(n_samples, len(files)), replace=False)

        vv_mask_vals, vv_bg_vals = [], []
        vh_mask_vals, vh_bg_vals = [], []

        for fname in chosen:
            vv, vh = load_image(os.path.join(cfg["images"], fname))
            mask = load_mask(os.path.join(cfg["masks"], fname))

            m = mask == 1
            bg = mask == 0
            n_s = min(3000, m.sum(), bg.sum())
            if n_s < 50:
                continue

            m_idx = np.random.choice(np.where(m.ravel())[0], n_s, replace=False)
            bg_idx = np.random.choice(np.where(bg.ravel())[0], n_s, replace=False)

            vv_mask_vals.append(vv.ravel()[m_idx])
            vv_bg_vals.append(vv.ravel()[bg_idx])
            vh_mask_vals.append(vh.ravel()[m_idx])
            vh_bg_vals.append(vh.ravel()[bg_idx])

        if not vv_mask_vals:
            continue

        vv_m = np.concatenate(vv_mask_vals)
        vv_b = np.concatenate(vv_bg_vals)
        vh_m = np.concatenate(vh_mask_vals)
        vh_b = np.concatenate(vh_bg_vals)

        axes[0, col_idx].hist(vv_b, bins=150, alpha=0.5, label="Фон", color="steelblue", density=True)
        axes[0, col_idx].hist(vv_m, bins=150, alpha=0.5, label="Маска", color="red", density=True)
        axes[0, col_idx].set_title(f"{title} — VV", fontsize=12, fontweight="bold")
        axes[0, col_idx].set_xlabel("Sigma0 VV (дБ)")
        axes[0, col_idx].legend()

        axes[1, col_idx].hist(vh_b, bins=150, alpha=0.5, label="Фон", color="steelblue", density=True)
        axes[1, col_idx].hist(vh_m, bins=150, alpha=0.5, label="Маска", color="red", density=True)
        axes[1, col_idx].set_title(f"{title} — VH", fontsize=12, fontweight="bold")
        axes[1, col_idx].set_xlabel("Sigma0 VH (дБ)")
        axes[1, col_idx].legend()

    axes[0, 0].set_ylabel("Плотность", fontsize=11)
    axes[1, 0].set_ylabel("Плотность", fontsize=11)

    plt.suptitle("Распределение пиксельных значений: маска vs фон", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "pixel_histograms_3class.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print("  [+] Пиксельные гистограммы (3 класса)")
    return path


# ──────────────────── Копируем аналитические графики ───────────────────

def copy_analysis_figures():
    """Копируем графики из figures/ в report/figures/ для встраивания."""
    import shutil
    paths = {}
    for fname in os.listdir(FIGURES_SRC):
        if fname.endswith(".png"):
            src = os.path.join(FIGURES_SRC, fname)
            dst = os.path.join(FIG_DIR, fname)
            shutil.copy2(src, dst)
            paths[fname.replace(".png", "")] = dst
    return paths


# ──────────────────── Генерация Markdown ───────────────────────────────

def generate_markdown(df, fig_paths):
    lines = []
    w = lines.append

    w("# Отчёт по датасету Sentinel-1 SAR Oil Spill")
    w("")
    w(f"> Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("> Источник: Zenodo — Part I (oil), Part II (lookalike, no_oil), Part III (test)")
    w("")

    # ── 1. Описание ──
    w("## 1. Описание датасета")
    w("")
    w("Датасет содержит SAR-снимки Sentinel-1 в формате GeoTIFF с двумя поляризационными "
      "каналами (VV и VH) в единицах Sigma0 (дБ). Каждому снимку соответствует бинарная маска.")
    w("")
    w("| Параметр | Значение |")
    w("|----------|----------|")
    w("| Размер SAR-снимка | 2048 × 2048 × 2 (VV, VH) |")
    w("| Размер маски | 2048 × 2048, бинарная |")
    w("| Формат | GeoTIFF, Sigma0 (дБ) |")
    w("| Спутник | Sentinel-1 |")
    w("| Лицензия | CC-BY-4.0 |")
    w("")

    # ── 2. Классы ──
    w("## 2. Распределение по классам")
    w("")
    w("| Класс | Описание | Кол-во снимков |")
    w("|-------|----------|---------------|")
    for cls, name in [("oil", "Реальное нефтяное загрязнение"),
                      ("lookalike", "Визуальный двойник (не нефть)"),
                      ("no_oil", "Чистая вода / без загрязнений")]:
        n = len(df[df["class"] == cls])
        w(f"| **{cls}** | {name} | {n} |")
    w(f"| **Всего** | | **{len(df)}** |")
    w("")

    if "01_class_distribution" in fig_paths:
        w(f"![Распределение по классам]({rel(fig_paths['01_class_distribution'])})")
        w("")
        w("*Рис. 1. Количество снимков по классам. Класс oil содержит почти вдвое больше "
          "данных, чем каждый из остальных.*")
        w("")

    # ── 3. Примеры ──
    w("## 3. Примеры изображений")
    w("")
    fig_num = 2
    for cls, name in [("oil", "Нефть"), ("lookalike", "Двойник (lookalike)"), ("no_oil", "Без нефти")]:
        key = f"samples_{cls}"
        if key in fig_paths:
            w(f"### 3.{fig_num-1}. Класс: {name}")
            w("")
            w(f"![Примеры {name}]({rel(fig_paths[key])})")
            w("")
            w(f"*Рис. {fig_num}. Примеры снимков класса «{name}»: VV, VH, маска и наложение.*")
            w("")
            fig_num += 1

    # ── 4. Экстремальные примеры ──
    has_extremes = any(f"extremes_{cls}" in fig_paths for cls in ["oil", "lookalike", "no_oil"])
    if has_extremes:
        w("## 4. Экстремальные примеры")
        w("")
        for cls, name in [("oil", "Нефть"), ("lookalike", "Двойник"), ("no_oil", "Без нефти")]:
            key = f"extremes_{cls}"
            if key in fig_paths:
                w(f"### Класс: {name}")
                w("")
                w(f"![Экстремальные {name}]({rel(fig_paths[key])})")
                w("")
                w(f"*Рис. {fig_num}. Снимки с наименьшей (верх) и наибольшей (низ) площадью маски для класса «{name}».*")
                w("")
                fig_num += 1

    # ── 5. Статистика масок ──
    w("## 5. Статистика масок по классам")
    w("")
    w("| Метрика | Нефть | Двойник | Без нефти |")
    w("|---------|-------|---------|-----------|")

    def fmt(series, fmt_str=".4%"):
        return f"{series:{fmt_str}}" if not pd.isna(series) else "—"

    for metric_name, col, fmt_str in [
        ("Средняя доля маски", "oil_ratio", ".4%"),
        ("Медианная доля маски", "oil_ratio", ".4%"),
        ("Мин. доля маски", "oil_ratio", ".4%"),
        ("Макс. доля маски", "oil_ratio", ".4%"),
        ("Ср. число компонент", "n_components", ".1f"),
        ("Медиана компонент", "n_components", ".0f"),
    ]:
        vals = []
        for cls in ["oil", "lookalike", "no_oil"]:
            sub = df[df["class"] == cls]
            if len(sub) == 0:
                vals.append("—")
                continue
            if "Средн" in metric_name:
                v = sub[col].mean()
            elif "Медиан" in metric_name:
                v = sub[col].median()
            elif "Мин" in metric_name:
                v = sub[col].min()
            elif "Макс" in metric_name:
                v = sub[col].max()
            else:
                v = sub[col].mean()

            if "%" in fmt_str:
                vals.append(f"{v:{fmt_str}}")
            else:
                vals.append(f"{v:{fmt_str}}")
        w(f"| {metric_name} | {vals[0]} | {vals[1]} | {vals[2]} |")
    w("")

    if "02_mask_ratio_by_class" in fig_paths:
        w(f"![Распределение масок]({rel(fig_paths['02_mask_ratio_by_class'])})")
        w("")
        w(f"*Рис. {fig_num}. Распределение доли маски по классам.*")
        w("")
        fig_num += 1

    if "04_components_by_class" in fig_paths:
        w(f"![Компоненты]({rel(fig_paths['04_components_by_class'])})")
        w("")
        w(f"*Рис. {fig_num}. Распределение числа связных компонент маски по классам.*")
        w("")
        fig_num += 1

    # ── 6. SAR-анализ ──
    has_sar = "vv_mean_all" in df.columns and df["vv_mean_all"].notna().any()
    if has_sar:
        w("## 6. SAR-характеристики")
        w("")

        w("### 6.1. Средние значения Sigma0 по классам")
        w("")
        w("| Класс | VV сред. (дБ) | VV std | VH сред. (дБ) | VH std |")
        w("|-------|-------------|--------|-------------|--------|")
        for cls, name in [("oil", "Нефть"), ("lookalike", "Двойник"), ("no_oil", "Без нефти")]:
            sub = df[df["class"] == cls].dropna(subset=["vv_mean_all"])
            if len(sub) == 0:
                continue
            w(f"| {name} | {sub['vv_mean_all'].mean():.2f} | {sub['vv_mean_all'].std():.2f} | "
              f"{sub['vh_mean_all'].mean():.2f} | {sub['vh_mean_all'].std():.2f} |")
        w("")

        if "03_sar_comparison" in fig_paths:
            w(f"![SAR сравнение]({rel(fig_paths['03_sar_comparison'])})")
            w("")
            w(f"*Рис. {fig_num}. Сравнение SAR-характеристик между классами.*")
            w("")
            fig_num += 1

        # Маскированные vs фон
        w("### 6.2. Маскированные области vs фон")
        w("")
        w("| Класс | Канал | Маска (дБ) | Фон (дБ) | Контраст (дБ) |")
        w("|-------|-------|-----------|----------|---------------|")
        for cls, name in [("oil", "Нефть"), ("lookalike", "Двойник")]:
            sub = df[df["class"] == cls]
            for ch in ["vv", "vh"]:
                m_col = f"{ch}_mean_oil"
                w_col = f"{ch}_mean_water"
                c_col = f"{ch}_contrast"
                valid = sub.dropna(subset=[m_col, w_col])
                if len(valid) > 0:
                    w(f"| {name} | {ch.upper()} | {valid[m_col].mean():.2f} | "
                      f"{valid[w_col].mean():.2f} | {valid[c_col].mean():.2f} |")
        w("")

        if "05_oil_vs_lookalike_sar" in fig_paths:
            w(f"![Нефть vs двойник]({rel(fig_paths['05_oil_vs_lookalike_sar'])})")
            w("")
            w(f"*Рис. {fig_num}. SAR-сигнатуры маскированных областей: нефть vs двойник.*")
            w("")
            fig_num += 1

        if "06_contrast_comparison" in fig_paths:
            w(f"![Контраст]({rel(fig_paths['06_contrast_comparison'])})")
            w("")
            w(f"*Рис. {fig_num}. Контраст (фон − маска) для классов oil и lookalike.*")
            w("")
            fig_num += 1

        # Пиксельные гистограммы
        if "pixel_histograms_3class" in fig_paths:
            w("### 6.3. Пиксельные распределения")
            w("")
            w(f"![Пиксельные гистограммы]({rel(fig_paths['pixel_histograms_3class'])})")
            w("")
            w(f"*Рис. {fig_num}. Распределения значений VV и VH для маскированных и фоновых пикселей по классам.*")
            w("")
            fig_num += 1

    # ── 7. Корреляции ──
    if "07_correlation_matrix" in fig_paths:
        w("## 7. Корреляционный анализ")
        w("")
        w(f"![Корреляции]({rel(fig_paths['07_correlation_matrix'])})")
        w("")
        w(f"*Рис. {fig_num}. Матрица корреляций числовых признаков.*")
        w("")
        fig_num += 1

    # ── 8. Выводы ──
    w("## 8. Ключевые выводы")
    w("")

    oil_df = df[df["class"] == "oil"]
    look_df = df[df["class"] == "lookalike"]
    nooil_df = df[df["class"] == "no_oil"]

    w(f"1. **Дисбаланс классов**: oil ({len(oil_df)}) значительно больше, чем "
      f"lookalike ({len(look_df)}) и no_oil ({len(nooil_df)}). "
      f"Необходим класс-балансирующий сэмплер.")

    if len(oil_df) > 0:
        w(f"2. **Дисбаланс масок (oil)**: медианная доля нефти всего "
          f"{oil_df['oil_ratio'].median():.2%} — сильная несбалансированность "
          f"для сегментации. Нужны Focal/Dice loss.")

    w(f"3. **Множественные пятна**: на {((df['n_components'] > 1).sum()/len(df)*100):.0f}% "
      f"снимков более одного маскированного региона — модель должна поддерживать "
      f"мультиобъектную сегментацию.")

    if has_sar:
        oil_vv = df[df["class"] == "oil"].dropna(subset=["vv_contrast"])
        look_vv = df[df["class"] == "lookalike"].dropna(subset=["vv_contrast"])
        if len(oil_vv) > 0 and len(look_vv) > 0:
            w(f"4. **Различие oil/lookalike по контрасту**: средний VV-контраст для нефти "
              f"{oil_vv['vv_contrast'].mean():.2f} дБ vs двойника "
              f"{look_vv['vv_contrast'].mean():.2f} дБ — модель должна учиться "
              f"различать эти классы.")

        w(f"5. **VH информативнее VV**: контраст в VH-канале выше, что подтверждает "
          f"целесообразность двухканального входа (VV + VH).")

    w(f"6. **Рекомендации для модели**: "
      f"двухканальный вход, поляриметрический фьюжн, "
      f"комбинированный loss (Focal + Dice), "
      f"MIL для классификации, "
      f"оверсэмплинг lookalike как hard negatives.")
    w("")

    return "\n".join(lines)


# ──────────────────── PDF-конвертация ──────────────────────────────────

CSS = """
@page { size: A4; margin: 2cm; }
body {
    font-family: "DejaVu Sans", "Noto Sans", Arial, sans-serif;
    font-size: 11pt; line-height: 1.5; color: #1a1a1a;
}
h1 { font-size: 20pt; color: #1a3c5e; border-bottom: 2px solid #1a3c5e;
     padding-bottom: 6px; margin-top: 30px; }
h2 { font-size: 15pt; color: #2a5f8f; border-bottom: 1px solid #ccc;
     padding-bottom: 4px; margin-top: 24px; }
h3 { font-size: 12pt; color: #3a7fbf; margin-top: 18px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 6px 10px; text-align: left; }
th { background-color: #e8f0f8; font-weight: bold; }
tr:nth-child(even) { background-color: #f9f9f9; }
img { max-width: 100%; display: block; margin: 16px auto; }
blockquote { border-left: 4px solid #2a5f8f; margin: 12px 0;
             padding: 8px 16px; background-color: #f0f6fc; font-style: italic; }
em { color: #555; font-size: 9.5pt; }
"""


def convert_to_pdf():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

    HTML(string=html_full, base_url=OUT_DIR).write_pdf(PDF_PATH)
    size_mb = os.path.getsize(PDF_PATH) / 1024 / 1024
    print(f"  PDF: {PDF_PATH} ({size_mb:.1f} МБ)")


# ──────────────────────────── main ────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Загрузка данных...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Загружено {len(df)} записей, классы: {df['class'].value_counts().to_dict()}")

    fig_paths = {}

    # Копируем аналитические графики
    print("Копирование аналитических графиков...")
    fig_paths.update(copy_analysis_figures())

    # Примеры изображений
    print("Генерация примеров изображений...")
    sample_paths = fig_samples_per_class(df)
    fig_paths.update(sample_paths)

    # Экстремальные примеры
    print("Генерация экстремальных примеров...")
    extreme_paths = fig_extremes_all_classes(df)
    fig_paths.update(extreme_paths)

    # Пиксельные гистограммы
    print("Генерация пиксельных гистограмм...")
    hist_path = fig_pixel_histograms_3class(df)
    fig_paths["pixel_histograms_3class"] = hist_path

    # Markdown
    print("Генерация отчёта...")
    md = generate_markdown(df, fig_paths)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  MD:  {REPORT_PATH}")

    # PDF
    print("Генерация PDF...")
    convert_to_pdf()

    print(f"\nГотово!")


if __name__ == "__main__":
    main()
