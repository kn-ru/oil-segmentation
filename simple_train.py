"""
Training loop для SimpleOilNet (tile-level).

Запуск:
    python simple_train.py
    python simple_train.py --epochs 50 --batch-size 4
    torchrun --nproc_per_node=2 simple_train.py --batch-size 2
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

from simple_config import SimpleConfig
from simple_dataset import TileDataset, ClassBalancedTileSampler
from simple_model import SimpleOilNet
from simple_losses import SimpleLoss
from dataset import (
    build_file_list, SARNormalizer, stratified_split,
    SARAugmenter, build_oil_patch_bank,
)
from metrics import MetricsAccumulator
from utils import set_seed, EMA, cosine_scheduler, setup_logger, save_checkpoint, count_parameters


def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--no-amp",     action="store_true")
    p.add_argument("--resume",     type=str,   default=None)
    p.add_argument("--grad-accum", type=int,   default=None)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema,
                    device, epoch, cfg, logger, rank=0, world_size=1):
    model.train()
    running = {"total": 0, "cls": 0, "seg": 0, "lovasz": 0, "dice": 0}
    n = 0
    accum = cfg.grad_accum_steps
    use_ddp = world_size > 1
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        image  = batch["image"].to(device, non_blocking=True)   # (B, 2, H, W)
        mask   = batch["mask"].to(device, non_blocking=True)    # (B, H, W)
        labels = batch["label"].to(device, non_blocking=True)

        is_sync = (i + 1) % accum == 0 or (i + 1) == len(loader)
        ctx = model.no_sync() if (use_ddp and not is_sync) else contextlib.nullcontext()

        with ctx:
            with autocast("cuda", enabled=cfg.use_amp):
                out = model(image)
                losses = criterion(out["class_logits"], labels,
                                   out["mask_logits"], mask)
                loss = losses["total"] / accum

            if cfg.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_sync:
            if cfg.use_amp:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if torch.isfinite(grad_norm):
                    optimizer.step()
            optimizer.zero_grad()
            if rank == 0:
                ema.update(model.module if use_ddp else model)

        for k in running:
            if k in losses:
                running[k] += losses[k].item()
        n += 1

        if rank == 0 and (i + 1) % cfg.log_every == 0:
            avg = {k: v / n for k, v in running.items()}
            logger.info(
                f"Epoch {epoch} [{i+1}/{len(loader)}] "
                f"loss={avg['total']:.4f} cls={avg['cls']:.4f} "
                f"seg={avg['seg']:.4f} (lovász={avg['lovasz']:.4f} dice={avg['dice']:.4f})"
            )

    avg = {k: v / max(1, n) for k, v in running.items()}
    if world_size > 1:
        keys = sorted(avg.keys())
        vals = torch.tensor([avg[k] for k in keys], device=device)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        vals /= world_size
        avg = {k: v.item() for k, v in zip(keys, vals)}
    return avg


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    metrics = MetricsAccumulator()
    running = {"total": 0, "cls": 0, "seg": 0}
    n = 0

    for batch in loader:
        image  = batch["image"].to(device, non_blocking=True)
        mask   = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast("cuda", enabled=cfg.use_amp):
            out = model(image)
            losses = criterion(out["class_logits"], labels,
                               out["mask_logits"], mask)

        for k in running:
            if k in losses:
                running[k] += losses[k].item()
        n += 1

        preds = out["class_logits"].argmax(1)
        metrics.update_classification(preds, labels)

        # mask_logits shape: (B, 1, H, W) → нужен (B, N, 1, H, W) для MetricsAccumulator
        # Оборачиваем в bag dim=1
        ml = out["mask_logits"].unsqueeze(1)  # (B, 1, 1, H, W)
        mt = mask.unsqueeze(1)               # (B, 1, H, W)
        metrics.update_segmentation(ml, mt, labels)

    avg_loss = {k: v / max(1, n) for k, v in running.items()}
    m = metrics.compute()
    m.update({f"loss_{k}": v for k, v in avg_loss.items()})
    return m


def main():
    args = parse_args()
    cfg = SimpleConfig()
    if args.epochs:     cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.lr = args.lr
    if args.no_amp:     cfg.use_amp = False
    if args.grad_accum: cfg.grad_accum_steps = args.grad_accum

    use_ddp_flag = is_ddp()
    if use_ddp_flag:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed + rank)
    logger = setup_logger(cfg.log_dir) if rank == 0 else None
    def log(msg):
        if rank == 0: logger.info(msg)

    eff_batch = cfg.batch_size * cfg.grad_accum_steps * world_size
    log(f"SimpleOilNet | device={device} | world_size={world_size} | eff_batch={eff_batch}")

    # ── Data ──
    entries = build_file_list(cfg.data_dir)
    train_entries, val_entries = stratified_split(entries, cfg.val_fraction, cfg.seed)
    log(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    norm_path = os.path.join(cfg.save_dir, "normalizer.json")
    normalizer = SARNormalizer()
    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        log(f"Loaded normalizer from {norm_path}")
    else:
        if rank == 0:
            log("Fitting normalizer...")
            normalizer.fit([e["image_path"] for e in train_entries], cfg.clip_percentiles)
            os.makedirs(cfg.save_dir, exist_ok=True)
            normalizer.save(norm_path)
        if use_ddp_flag:
            dist.barrier()
            if rank != 0: normalizer.load(norm_path)

    oil_entries = [e for e in train_entries if e["class_name"] == "oil"]
    oil_bank = build_oil_patch_bank(oil_entries, normalizer, max_images=100)
    log(f"Patch bank: {len(oil_bank['vv']) if oil_bank else 0} patches")

    augmenter = SARAugmenter(speckle_prob=0.5, radio_prob=0.5,
                              gauss_prob=0.3, copypaste_prob=0.5)

    train_ds = TileDataset(
        train_entries, normalizer, cfg.tile_size,
        augment=True, augmenter=augmenter, oil_patch_bank=oil_bank,
        force_oil_crop_prob=cfg.force_oil_crop_prob,
        force_oil_min_ratio=cfg.force_oil_min_ratio,
    )
    val_ds = TileDataset(val_entries, normalizer, cfg.tile_size, augment=False)

    if use_ddp_flag:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        val_sampler   = None
    else:
        train_sampler = ClassBalancedTileSampler(train_ds, cfg.lookalike_oversample, cfg.seed)
        val_sampler   = None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, sampler=val_sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False,
    )

    # ── Model ──
    model = SimpleOilNet(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        fpn_channels=cfg.decoder_channels[0],
        num_classes=cfg.num_classes,
    ).to(device)

    if use_ddp_flag:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw_model = model.module if use_ddp_flag else model
    if rank == 0:
        p = count_parameters(raw_model)
        log(f"Parameters: {p['trainable_M']:.1f}M trainable")

    criterion = SimpleLoss(
        seg_weight=cfg.seg_weight,
        lovasz_weight=cfg.lovasz_weight,
        dice_weight=cfg.dice_weight,
        label_smoothing=cfg.label_smoothing,
        class_weights=cfg.class_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    scheduler = cosine_scheduler(optimizer, cfg.lr, cfg.warmup_epochs, cfg.epochs)
    scaler = GradScaler("cuda", enabled=cfg.use_amp)
    ema = EMA(raw_model, cfg.ema_decay) if rank == 0 else None

    start_epoch = 0
    best_score = 0

    if args.resume:
        from utils import load_checkpoint
        start_epoch = load_checkpoint(args.resume, raw_model, optimizer, scheduler, ema)
        log(f"Resumed from epoch {start_epoch}")

    log("Starting training...")
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        if use_ddp_flag and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        elif hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            ema, device, epoch, cfg, logger, rank, world_size,
        )
        scheduler.step()

        if rank == 0:
            log(f"Epoch {epoch} done in {time.time()-t0:.0f}s | "
                f"loss={losses['total']:.4f} seg={losses['seg']:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if rank == 0 and (epoch + 1) % cfg.val_every == 0:
            m = validate(ema.shadow, val_loader, criterion, device, cfg)
            log(f"Val: macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
                f"oil_iou={m['oil_iou']:.4f} oil_dice={m['oil_dice']:.4f} "
                f"loss={m['loss_total']:.4f}")
            log(f"  Recall: oil={m['recall_oil']:.3f} look={m['recall_lookalike']:.3f} "
                f"nooil={m['recall_no_oil']:.3f}")
            log(f"  FP: look={m['fp_area_lookalike']:.6f} nooil={m['fp_area_no_oil']:.6f}")

            score = m["macro_f1"] + m["oil_iou"]   # IoU как основная метрика
            if score > best_score and not (m["loss_total"] != m["loss_total"]):
                best_score = score
                save_checkpoint(
                    os.path.join(cfg.save_dir, "best.pt"),
                    epoch, raw_model, optimizer, scheduler, ema, m,
                )
                log(f"  New best! score={score:.4f}")

        if rank == 0 and (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.save_dir, f"epoch_{epoch}.pt"),
                epoch, raw_model, optimizer, scheduler, ema,
            )

        if use_ddp_flag:
            dist.barrier()

    if rank == 0:
        save_checkpoint(os.path.join(cfg.save_dir, "last.pt"),
                        cfg.epochs - 1, raw_model, optimizer, scheduler, ema)
        log("Done.")

    if use_ddp_flag:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
