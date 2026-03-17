# DualPolMAFUformerMIL — SAR Oil Spill Detection

Модель для обнаружения и сегментации нефтяных разливов по спутниковым снимкам Sentinel-1 SAR.

## Архитектура

**DualPolMAFUformerMIL** — гибридная модель с:
- Двумя параллельными стемами для VV и VH каналов
- Поляриметрическим фьюжн-блоком (gated attention)
- Backbone ConvNeXt-Small с bottleneck (Window Self-Attention)
- U-образным декодером с SAFB (Skip Attention Fusion Blocks) и **deep supervision**
- **Multi-scale** сегментационной головой
- MIL (Multiple Instance Learning) классификатором на Transformer-encoder

```
VV ──→ Stem ──┐
              ├──→ PolarimetricFusion ──→ ConvNeXt Backbone ──→ Bottleneck
VH ──→ Stem ──┘                                                     │
                                                            U-Decoder + SAFB
                                                                    │
                                             ┌──────────────────────┤
                                      SegHead (multi-scale)   TileEmbedding
                                             │                      │
                                        Mask logits      MIL BagClassifier
                                                                    │
                                                           Class logits (3)
```

## Датасет

[Zenodo Sentinel-1 SAR Oil Spill Dataset](https://zenodo.org/records/8346860)

| Набор | Класс | Изображений |
|-------|-------|-------------|
| Train/Val | oil | 1200 |
| Train/Val | lookalike | 685 |
| Train/Val | no_oil | 685 |
| Test | oil / lookalike / no_oil | 150 × 3 |

- Размер снимков: **2048 × 2048 × 2** (VV, VH), Sigma0 в дБ
- Маски: бинарные 2048 × 2048
- Медианная доля нефти: **1.76%** пикселей (сильный дисбаланс)

## Структура проекта

```
oil_detection/
├── config.py           # Конфигурация (dataclasses)
├── dataset.py          # Dataset, тайлинг, SAR аугментации, copy-paste
├── losses.py           # FocalLoss + DiceLoss + BoundaryLoss + OHEM
├── metrics.py          # macro-F1, IoU, Dice, confusion matrix, FP area
├── train.py            # Training loop (AMP, EMA, grad accumulation)
├── inference.py        # Sliding-window inference + TTA
├── utils.py            # Seed, EMA, cosine scheduler, checkpoints
├── model/
│   ├── stems.py        # Dual-pol shallow stems
│   ├── fusion.py       # Polarimetric fusion block
│   ├── backbone.py     # ConvNeXt backbone (timm)
│   ├── bottleneck.py   # Window Self-Attention + DW-Conv MLP
│   ├── decoder.py      # U-Decoder + SAFB + deep supervision
│   ├── heads.py        # SegHead (multi-scale) + TileEmbed + MIL classifier
│   └── dualpol.py      # Полная сборка модели
├── analyze_data.py     # Анализ датасета (статистика, графики)
├── visualize_data.py   # Визуализация примеров
├── generate_report.py  # Генерация MD-отчёта с графиками
└── md_to_pdf.py        # Конвертация отчёта в PDF
```

## Установка

```bash
pip install torch torchvision timm rasterio numpy pandas matplotlib seaborn scipy weasyprint markdown
```

## Подготовка данных

```
/media/knru/DataLake/OIL/
├── Oil/                # train/val нефть (1200 снимков)
├── Mask_oil/
├── Lookalike/          # train/val двойники (685)
├── Mask_lookalike/
├── No_oil/             # train/val без нефти (685)
├── Mask_no_oil/
├── Images/             # test (150 × 3 классов)
│   ├── Oil/
│   ├── Lookalike/
│   └── No oil/
└── Mask/               # test маски
    ├── Oil/
    ├── Lookalike/
    └── No oil/
```

## Обучение

```bash
# Полное обучение (150 эпох)
python3 train.py

# На GPU с ограниченной VRAM (< 24 GB)
python3 train.py --batch-size 1 --grad-accum 8

# С меньшим backbone
python3 train.py --backbone convnext_tiny --batch-size 1 --grad-accum 4

# Продолжить с чекпоинта
python3 train.py --resume checkpoints/best.pt
```

Чекпоинты сохраняются в `checkpoints/`, логи в `runs/`.

## Инференс

```bash
# Один файл
python3 inference.py input.tif --checkpoint checkpoints/best.pt --output-dir predictions/

# Папка с файлами (с TTA)
python3 inference.py /path/to/images/ --checkpoint checkpoints/best.pt

# Без TTA (быстрее)
python3 inference.py input.tif --checkpoint checkpoints/best.pt --no-tta
```

## Анализ датасета

```bash
# Полный анализ + графики
python3 analyze_data.py

# Только маски (быстрее)
python3 analyze_data.py --no-sar

# Генерация отчёта (MD + PDF)
python3 generate_report.py
```

## Ключевые решения

| Проблема | Решение |
|----------|---------|
| Нефть ~2% площади | Focal loss (γ=2, α=0.75) + Dice + Boundary loss + OHEM |
| Мелкие пятна | Deep supervision + multi-scale seg head |
| Дисбаланс классов | Class weights + ClassBalancedSampler + lookalike oversampling |
| Малое количество нефти | Force oil crop (50%) + copy-paste аугментация |
| SAR-специфика | Speckle noise + radiometric shift/scale аугментации |
| Lookalike hard negatives | 1.5× oversampling lookalike в сэмплере |
| Стабильность обучения | EMA (0.999) + gradient accumulation (eff batch 8–16) |

## Loss

```
L_total = L_cls + 0.7 * L_seg + 0.4 * L_ds

L_cls = CrossEntropy(label_smoothing=0.05, class_weights=[0.71, 1.25, 1.25])
L_seg = 0.5 * FocalLoss + 0.3 * DiceLoss + 0.2 * BoundaryLoss  (+ OHEM top 70%)
L_ds  = weighted sum of auxiliary seg losses (deep supervision)
```

## Метрики

- **Классификация**: macro-F1, per-class recall, accuracy, confusion matrix
- **Сегментация**: oil IoU, oil Dice
- **Negative control**: FP oil area на lookalike/no_oil изображениях

## Параметры модели

- **72.24M** параметров (ConvNeXt-Small backbone)
- Вход тайла: 512 × 512 × 2 (VV + VH)
- Bag: 16 тайлов на изображение (4 × 4 сетка)
- Sliding window инференс: окно 512, stride 384
