"""
Training loop для DualPolMAFUformerMIL с поддержкой DDP.

Запуск на одной GPU:
    python train.py --batch-size 1 --grad-accum 8

Запуск DDP на N GPU (torchrun):
    torchrun --nproc_per_node=N train.py --batch-size 1 --grad-accum 4

Запуск DDP на 2 узлах:
    torchrun --nnodes=2 --nproc_per_node=N --rdzv_id=42 \\
             --rdzv_backend=c10d --rdzv_endpoint=HOST:PORT train.py
"""

import os
import argparse
import contextlib
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

from config import Config
from dataset import (
    build_file_list, OilSpillBagDataset, SARNormalizer,
    ClassBalancedSampler, stratified_split,
    SARAugmenter, build_oil_patch_bank,
)
from model import DualPolMAFUformerMIL
from losses import TotalLoss
from metrics import MetricsAccumulator
from utils import (
    set_seed, EMA, cosine_scheduler, setup_logger,
    save_checkpoint, load_checkpoint, count_parameters,
)


# ──────────────────── DDP helpers ──────────────────────────────────────

def setup_ddp():
    """Инициализация процессной группы из переменных окружения torchrun."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_ddp_available():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def reduce_dict(d: dict, world_size: int) -> dict:
    """Усреднить словарь loss-значений по всем GPU."""
    keys = sorted(d.keys())
    values = torch.tensor([d[k] for k in keys], dtype=torch.float32,
                          device=torch.cuda.current_device())
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return {k: v.item() for k, v in zip(keys, values)}


# ──────────────────── ClassBalancedDistributedSampler ──────────────────

class ClassBalancedDistributedSampler(DistributedSampler):
    """DistributedSampler с балансировкой классов и оверсэмплингом lookalike.

    Совмещает ClassBalancedSampler логику с DDP-aware разбивкой на ранги.
    """

    def __init__(self, entries, lookalike_oversample=1.5, seed=42,
                 num_replicas=None, rank=None, drop_last=True):
        # DistributedSampler инициализируем с фиктивным dataset (len = наш total)
        super().__init__(
            dataset=entries,  # используем как dummy для len()
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=drop_last,
        )
        self.entries = entries
        self.lookalike_oversample = lookalike_oversample
        self._build_indices()

    def _build_indices(self):
        import numpy as np
        # Группировка по классам
        class_indices = {}
        for i, e in enumerate(self.entries):
            cls = e["class_name"]
            if cls not in class_indices:
                class_indices[cls] = []
            class_indices[cls].append(i)

        max_count = max(len(v) for v in class_indices.values())
        self._all_indices = []
        for cls, idxs in class_indices.items():
            n = int(max_count * self.lookalike_oversample) if cls == "lookalike" else max_count
            rng = np.random.RandomState(self.seed)
            sampled = rng.choice(idxs, size=n, replace=True).tolist()
            self._all_indices.extend(sampled)

        self.total_size = (len(self._all_indices) // self.num_replicas) * self.num_replicas

    def __iter__(self):
        import numpy as np
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = self._all_indices.copy()
        rng.shuffle(indices)

        # Обрезать до кратного num_replicas
        indices = indices[:self.total_size]

        # Распределить по рангам
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.total_size // self.num_replicas


# ──────────────────── Train one epoch ──────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: TotalLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
    epoch: int,
    cfg: Config,
    logger,
    rank: int = 0,
    world_size: int = 1,
):
    model.train()
    running = {"total": 0, "cls": 0, "seg": 0, "ds": 0}
    n_batches = 0
    accum_steps = cfg.train.grad_accum_steps
    is_ddp = world_size > 1

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        vv    = batch["vv"].to(device, non_blocking=True)
        vh    = batch["vh"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # В DDP синхронизировать градиенты только на последнем шаге аккумуляции
        is_sync_step = (
            (batch_idx + 1) % accum_steps == 0
            or (batch_idx + 1) == len(dataloader)
        )

        sync_ctx = (
            model.no_sync()
            if (is_ddp and not is_sync_step)
            else contextlib.nullcontext()
        )

        with sync_ctx:
            with autocast(device_type="cuda", enabled=cfg.train.use_amp):
                outputs = model(vv, vh)
                loss_dict = criterion(
                    outputs["class_logits"], labels,
                    outputs["mask_logits"], masks,
                    aux_logits=outputs.get("aux_logits"),
                    ds_weights=outputs.get("ds_weights"),
                )
                loss = loss_dict["total"] / accum_steps

            if cfg.train.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_sync_step:
            if cfg.train.use_amp:
                scaler.unscale_(optimizer)
                # Проверка на NaN/Inf в градиентах
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                else:
                    if rank == 0:
                        logger.warning(f"Epoch {epoch} batch {batch_idx}: "
                                       f"NaN/Inf grad detected, skipping step")
                scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                if torch.isfinite(grad_norm):
                    optimizer.step()

            optimizer.zero_grad()
            # EMA только на rank 0 и только если шаг был валидным
            if rank == 0 and torch.isfinite(grad_norm):
                ema.update(model.module if is_ddp else model)

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k].item()
        n_batches += 1

        if rank == 0 and (batch_idx + 1) % cfg.train.log_every == 0:
            avg = {k: v / n_batches for k, v in running.items()}
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={avg['total']:.4f} cls={avg['cls']:.4f} "
                f"seg={avg['seg']:.4f} ds={avg['ds']:.4f}"
            )

    avg = {k: v / max(1, n_batches) for k, v in running.items()}

    # Усреднить метрики по всем GPU
    if is_ddp:
        avg = reduce_dict(avg, world_size)

    return avg


# ──────────────────── Validation ───────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: TotalLoss,
    device: torch.device,
    cfg: Config,
    rank: int = 0,
    world_size: int = 1,
):
    model.eval()
    metrics = MetricsAccumulator()
    running = {"total": 0, "cls": 0, "seg": 0, "ds": 0}
    n_batches = 0

    for batch in dataloader:
        vv    = batch["vv"].to(device, non_blocking=True)
        vh    = batch["vh"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=cfg.train.use_amp):
            outputs = model(vv, vh)
            loss_dict = criterion(
                outputs["class_logits"], labels,
                outputs["mask_logits"], masks,
                aux_logits=outputs.get("aux_logits"),
                ds_weights=outputs.get("ds_weights"),
            )

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k].item()
        n_batches += 1

        preds = outputs["class_logits"].argmax(dim=1)
        metrics.update_classification(preds, labels)
        metrics.update_segmentation(outputs["mask_logits"], masks, labels)

    avg_loss = {k: v / max(1, n_batches) for k, v in running.items()}

    if world_size > 1:
        avg_loss = reduce_dict(avg_loss, world_size)

    metric_vals = metrics.compute()
    metric_vals.update({f"loss_{k}": v for k, v in avg_loss.items()})

    return metric_vals, metrics.confusion_matrix()


# ──────────────────── Main ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train DualPolMAFUformerMIL (DDP)")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--no-amp",     action="store_true")
    parser.add_argument("--resume",     type=str,   default=None)
    parser.add_argument("--backbone",   type=str,   default=None)
    parser.add_argument("--grad-accum", type=int,   default=None)
    parser.add_argument("--sync-bn",    action="store_true",
                        help="Convert BN to SyncBN (DDP multi-GPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── DDP setup ──
    use_ddp = is_ddp_available()
    if use_ddp:
        rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    if args.epochs:     cfg.train.epochs           = args.epochs
    if args.batch_size: cfg.data.batch_size         = args.batch_size
    if args.lr:         cfg.train.lr               = args.lr
    if args.no_amp:     cfg.train.use_amp           = False
    if args.backbone:   cfg.model.backbone_name     = args.backbone
    if args.grad_accum: cfg.train.grad_accum_steps  = args.grad_accum

    set_seed(cfg.seed + rank)

    # Логгер только на rank 0
    logger = setup_logger(cfg.train.log_dir) if rank == 0 else None

    def log(msg):
        if rank == 0:
            logger.info(msg)

    eff_batch = cfg.data.batch_size * cfg.train.grad_accum_steps * world_size
    log(f"DDP: {'ON' if use_ddp else 'OFF'} | world_size={world_size} | rank={rank} | device={device}")
    log(f"Config: epochs={cfg.train.epochs}, bs/gpu={cfg.data.batch_size}, "
        f"grad_accum={cfg.train.grad_accum_steps} (eff_batch={eff_batch}), "
        f"lr={cfg.train.lr}, amp={cfg.train.use_amp}")
    log(f"Loss: seg_w={cfg.train.seg_weight}, ds_w={cfg.train.ds_weight}, "
        f"focal={cfg.train.focal_weight}, dice={cfg.train.dice_weight}, "
        f"boundary={cfg.train.boundary_weight}, ohem={cfg.train.ohem_ratio}")
    log(f"Class weights: {cfg.train.class_weights}")

    # ── Data ──
    log("Building file list...")
    entries = build_file_list(cfg.data.data_dir)
    log(f"Total entries: {len(entries)}")

    train_entries, val_entries = stratified_split(entries, cfg.data.val_fraction, cfg.seed)
    log(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    # Normalizer — fit только на rank 0, остальные ждут
    norm_path = os.path.join(cfg.train.save_dir, "normalizer.json")
    normalizer = SARNormalizer()
    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        log(f"Loaded normalizer from {norm_path}")
    else:
        if rank == 0:
            log("Fitting normalizer on train data...")
            train_img_paths = [e["image_path"] for e in train_entries]
            normalizer.fit(train_img_paths, cfg.data.clip_percentiles)
            os.makedirs(cfg.train.save_dir, exist_ok=True)
            normalizer.save(norm_path)
            log(f"Saved normalizer to {norm_path}")
        if use_ddp:
            dist.barrier()  # остальные ранги ждут пока rank 0 сохранит
            if rank != 0:
                normalizer.load(norm_path)

    # Oil patch bank — строим на каждом GPU независимо (данные в CPU памяти)
    log("Building oil patch bank for copy-paste augmentation...")
    oil_train_entries = [e for e in train_entries if e["class_name"] == "oil"]
    oil_patch_bank = build_oil_patch_bank(
        oil_train_entries, normalizer, max_images=100, patches_per_image=5)
    n_patches = len(oil_patch_bank["vv"]) if oil_patch_bank else 0
    log(f"  Patch bank: {n_patches} patches from oil images")

    augmenter = SARAugmenter(
        speckle_prob=0.5, speckle_looks=4,
        radio_prob=0.5, radio_shift=0.3, radio_scale=0.15,
        gauss_prob=0.3, gauss_std=0.1,
        copypaste_prob=0.5, copypaste_patches=3,
    )

    train_ds = OilSpillBagDataset(
        train_entries, normalizer, cfg.data.tile_size,
        force_oil_crop_prob=cfg.data.force_oil_crop_prob,
        force_oil_min_ratio=cfg.data.force_oil_min_ratio,
        augment=True,
        augmenter=augmenter,
        oil_patch_bank=oil_patch_bank,
    )
    val_ds = OilSpillBagDataset(val_entries, normalizer, cfg.data.tile_size, augment=False)

    # Сэмплер
    if use_ddp:
        train_sampler = ClassBalancedDistributedSampler(
            train_entries,
            lookalike_oversample=cfg.train.lookalike_oversample,
            seed=cfg.seed,
            num_replicas=world_size,
            rank=rank,
        )
        # Валидация только на rank 0 (полный val set, без шардирования)
        val_sampler = None
    else:
        train_sampler = ClassBalancedSampler(
            train_entries, cfg.train.lookalike_oversample, cfg.seed)
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, sampler=train_sampler,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, sampler=val_sampler, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    # ── Model ──
    log("Building model...")
    model = DualPolMAFUformerMIL.from_config(cfg).to(device)

    if args.sync_bn and use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        log("Converted BatchNorm -> SyncBatchNorm")

    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw_model = model.module if use_ddp else model

    if rank == 0:
        params = count_parameters(raw_model)
        log(f"Parameters: {params['trainable_M']:.2f}M trainable / "
            f"{params['total_M']:.2f}M total")

    # ── Loss ──
    criterion = TotalLoss(
        seg_weight=cfg.train.seg_weight,
        focal_gamma=cfg.train.focal_gamma,
        focal_alpha=cfg.train.focal_alpha,
        focal_weight=cfg.train.focal_weight,
        dice_weight=cfg.train.dice_weight,
        boundary_weight=cfg.train.boundary_weight,
        ohem_ratio=cfg.train.ohem_ratio,
        label_smoothing=cfg.train.label_smoothing,
        num_classes=cfg.data.num_classes,
        class_weights=cfg.train.class_weights,
        ds_weight=cfg.train.ds_weight,
    ).to(device)

    # ── Optimizer, Scheduler ──
    # LR НЕ масштабируем: effective batch = bs * grad_accum * world_size
    # При DDP с grad_accum пользователь уже контролирует eff batch через --grad-accum
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = cosine_scheduler(
        optimizer, cfg.train.lr, cfg.train.warmup_epochs, cfg.train.epochs)

    scaler = GradScaler("cuda", enabled=cfg.train.use_amp)
    ema = EMA(raw_model, cfg.train.ema_decay) if rank == 0 else None

    start_epoch = 0
    best_metric = 0

    if args.resume:
        start_epoch = load_checkpoint(args.resume, raw_model, optimizer, scheduler,
                                      ema if rank == 0 else None)
        if use_ddp:
            dist.barrier()
        log(f"Resumed from epoch {start_epoch}")

    # ── Training Loop ──
    log("Starting training...")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        train_sampler.set_epoch(epoch)

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            ema, device, epoch, cfg, logger, rank, world_size,
        )
        scheduler.step()

        if rank == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            log(f"Epoch {epoch} done in {elapsed:.0f}s | "
                f"loss={train_losses['total']:.4f} ds={train_losses['ds']:.4f} | "
                f"lr={lr:.2e}")

        # Validation — только rank 0 считает метрики через EMA
        if rank == 0 and (epoch + 1) % cfg.train.val_every == 0:
            val_metrics, conf_mat = validate(
                ema.shadow, val_loader, criterion, device, cfg, rank=0, world_size=1)

            log(f"Val: macro_f1={val_metrics['macro_f1']:.4f} "
                f"acc={val_metrics['accuracy']:.4f} "
                f"oil_iou={val_metrics['oil_iou']:.4f} "
                f"oil_dice={val_metrics['oil_dice']:.4f} "
                f"loss={val_metrics['loss_total']:.4f}")
            log(f"  Per-class recall: "
                f"oil={val_metrics['recall_oil']:.3f} "
                f"look={val_metrics['recall_lookalike']:.3f} "
                f"nooil={val_metrics['recall_no_oil']:.3f}")
            log(f"  FP area: "
                f"look={val_metrics['fp_area_lookalike']:.6f} "
                f"nooil={val_metrics['fp_area_no_oil']:.6f}")

            score = val_metrics["macro_f1"] + val_metrics["oil_dice"]
            if score > best_metric and not (val_metrics["loss_total"] != val_metrics["loss_total"]):
                best_metric = score
                save_checkpoint(
                    os.path.join(cfg.train.save_dir, "best.pt"),
                    epoch, raw_model, optimizer, scheduler, ema, val_metrics,
                )
                log(f"  New best! score={score:.4f}")

        if rank == 0 and (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.train.save_dir, f"epoch_{epoch}.pt"),
                epoch, raw_model, optimizer, scheduler, ema,
            )

        if use_ddp:
            dist.barrier()

    if rank == 0:
        save_checkpoint(
            os.path.join(cfg.train.save_dir, "last.pt"),
            cfg.train.epochs - 1, raw_model, optimizer, scheduler, ema,
        )
        log("Training complete.")

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
