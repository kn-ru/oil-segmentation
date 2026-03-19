"""
Seg-only training: только сегментация, без classification loss.

Запуск:
    python seg_train.py --batch-size 16 --lr 3e-4
    torchrun --nproc_per_node=2 seg_train.py --batch-size 16 --lr 3e-4
"""

import os
import argparse
import contextlib
import time

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

from simple_config import SimpleConfig
from simple_dataset import TileDataset, ClassBalancedTileSampler
from seg_model import SegOnlyNet
from seg_losses import SegOnlyLoss
from dataset import (
    build_file_list, SARNormalizer, stratified_split,
    SARAugmenter, build_oil_patch_bank,
)
from utils import set_seed, EMA, cosine_scheduler, setup_logger, save_checkpoint, count_parameters

import numpy as np


def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--no-amp",     action="store_true")
    p.add_argument("--resume",     type=str,   default=None)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--data-dir",   type=str,   default=None)
    p.add_argument("--oil-only",   action="store_true", default=True,
                   help="Train only on oil images (default: True)")
    p.add_argument("--all-classes", action="store_true",
                   help="Train on all classes (disable oil-only)")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema,
                    device, epoch, cfg, logger, rank=0, world_size=1,
                    grad_accum=4, grad_clip=1.0, use_amp=True):
    model.train()
    running = {"total": 0, "lovasz": 0, "dice": 0, "bce": 0}
    n = 0
    use_ddp = world_size > 1
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)

        is_sync = (i + 1) % grad_accum == 0 or (i + 1) == len(loader)
        ctx = model.no_sync() if (use_ddp and not is_sync) else contextlib.nullcontext()

        with ctx:
            with autocast("cuda", enabled=use_amp):
                mask_logits = model(image)
                losses = criterion(mask_logits, mask)
                loss = losses["total"] / grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_sync:
            if use_amp:
                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isfinite(gn):
                    scaler.step(optimizer)
                scaler.update()
            else:
                gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isfinite(gn):
                    optimizer.step()
            optimizer.zero_grad()
            if rank == 0 and ema is not None:
                ema.update(model.module if use_ddp else model)

        for k in running:
            if k in losses:
                running[k] += losses[k].item()
        n += 1

        if rank == 0 and logger and (i + 1) % 20 == 0:
            avg = {k: v / n for k, v in running.items()}
            logger.info(
                f"Epoch {epoch} [{i+1}/{len(loader)}] "
                f"loss={avg['total']:.4f} "
                f"(lovász={avg['lovasz']:.4f} dice={avg['dice']:.4f} bce={avg['bce']:.4f})"
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
def validate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_iou = 0
    total_dice = 0
    total_loss = 0
    n = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            mask_logits = model(image)
            losses = criterion(mask_logits, mask)

        total_loss += losses["total"].item()

        # Compute IoU and Dice per batch
        probs = torch.sigmoid(mask_logits.squeeze(1))
        pred_binary = (probs > 0.5).float()
        targets = mask.float()

        inter = (pred_binary * targets).sum(dim=(1, 2))
        union = pred_binary.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        dice = (2 * inter + 1e-6) / (pred_binary.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + 1e-6)

        # Only count samples that have oil in GT
        has_oil = targets.sum(dim=(1, 2)) > 0
        if has_oil.any():
            total_iou += iou[has_oil].sum().item()
            total_dice += dice[has_oil].sum().item()
            n += has_oil.sum().item()

    avg_iou  = total_iou / max(1, n)
    avg_dice = total_dice / max(1, n)
    avg_loss = total_loss / max(1, len(loader))

    return {"oil_iou": avg_iou, "oil_dice": avg_dice, "loss": avg_loss, "n_oil": n}


def main():
    args = parse_args()
    cfg = SimpleConfig()
    if args.data_dir:   cfg.data_dir = args.data_dir

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
    save_dir = "checkpoints_seg"
    log_dir  = "runs_seg"
    logger = setup_logger(log_dir) if rank == 0 else None
    def log(msg):
        if rank == 0: logger.info(msg)

    eff_batch = args.batch_size * args.grad_accum * world_size
    log(f"SegOnlyNet | device={device} | world_size={world_size} | "
        f"bs={args.batch_size} accum={args.grad_accum} eff_batch={eff_batch} lr={args.lr}")

    # ── Data ──
    oil_only = args.oil_only and not args.all_classes
    entries = build_file_list(cfg.data_dir)
    train_entries, val_entries = stratified_split(entries, cfg.val_fraction, cfg.seed)

    if oil_only:
        train_entries = [e for e in train_entries if e["class_name"] == "oil"]
        val_entries   = [e for e in val_entries   if e["class_name"] == "oil"]
        log(f"OIL-ONLY mode: Train: {len(train_entries)}, Val: {len(val_entries)}")
    else:
        log(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    norm_path = os.path.join(save_dir, "normalizer.json")
    normalizer = SARNormalizer()
    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        log(f"Loaded normalizer from {norm_path}")
    elif os.path.exists(os.path.join(cfg.save_dir, "normalizer.json")):
        normalizer.load(os.path.join(cfg.save_dir, "normalizer.json"))
        log(f"Loaded normalizer from {cfg.save_dir}")
    else:
        if rank == 0:
            log("Fitting normalizer...")
            normalizer.fit([e["image_path"] for e in train_entries], cfg.clip_percentiles)
            os.makedirs(save_dir, exist_ok=True)
            normalizer.save(norm_path)
        if use_ddp_flag:
            dist.barrier()
            if rank != 0: normalizer.load(norm_path)

    # Copy-paste не нужен в oil-only (все данные уже oil)
    if oil_only:
        oil_bank = None
        augmenter = SARAugmenter(speckle_prob=0.5, radio_prob=0.5,
                                  gauss_prob=0.3, copypaste_prob=0.0)
    else:
        oil_entries_for_bank = [e for e in train_entries if e["class_name"] == "oil"]
        oil_bank = build_oil_patch_bank(oil_entries_for_bank, normalizer, max_images=100)
        log(f"Patch bank: {len(oil_bank['vv']) if oil_bank else 0} patches")
        augmenter = SARAugmenter(speckle_prob=0.5, radio_prob=0.5,
                                  gauss_prob=0.3, copypaste_prob=0.5)

    train_ds = TileDataset(
        train_entries, normalizer, cfg.tile_size,
        augment=True, augmenter=augmenter, oil_patch_bank=oil_bank,
        force_oil_crop_prob=cfg.force_oil_crop_prob if not oil_only else 0.0,
        force_oil_min_ratio=cfg.force_oil_min_ratio,
    )
    val_ds = TileDataset(val_entries, normalizer, cfg.tile_size, augment=False)

    if use_ddp_flag:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    else:
        from torch.utils.data import RandomSampler
        train_sampler = RandomSampler(train_ds)

    mp_ctx = 'spawn' if cfg.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True,
        multiprocessing_context=mp_ctx,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False,
        multiprocessing_context=mp_ctx,
    )

    # ── Model ──
    model = SegOnlyNet(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        fpn_channels=256,
    ).to(device)

    if use_ddp_flag:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw_model = model.module if use_ddp_flag else model
    if rank == 0:
        p = count_parameters(raw_model)
        log(f"Parameters: {p['trainable_M']:.1f}M trainable")

    criterion = SegOnlyLoss(lovasz_weight=0.5, dice_weight=0.3, bce_weight=0.2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=cfg.weight_decay)
    scheduler = cosine_scheduler(optimizer, args.lr, cfg.warmup_epochs, args.epochs)
    scaler = GradScaler("cuda", enabled=cfg.use_amp)
    ema = EMA(raw_model, cfg.ema_decay) if rank == 0 else None

    start_epoch = 0
    best_iou = 0

    log("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if use_ddp_flag and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        elif hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            ema, device, epoch, cfg, logger, rank, world_size,
            grad_accum=args.grad_accum, grad_clip=cfg.grad_clip,
            use_amp=cfg.use_amp,
        )
        scheduler.step()

        if rank == 0:
            log(f"Epoch {epoch} done in {time.time()-t0:.0f}s | "
                f"loss={losses['total']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if rank == 0 and (epoch + 1) % cfg.val_every == 0:
            m = validate(ema.shadow, val_loader, criterion, device, cfg.use_amp)
            log(f"Val: oil_iou={m['oil_iou']:.4f} oil_dice={m['oil_dice']:.4f} "
                f"loss={m['loss']:.4f} (n_oil={m['n_oil']})")

            if m["oil_iou"] > best_iou:
                best_iou = m["oil_iou"]
                os.makedirs(save_dir, exist_ok=True)
                save_checkpoint(
                    os.path.join(save_dir, "best.pt"),
                    epoch, raw_model, optimizer, scheduler, ema, m,
                )
                log(f"  New best! oil_iou={best_iou:.4f}")

        if rank == 0 and (epoch + 1) % cfg.save_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(save_dir, f"epoch_{epoch}.pt"),
                epoch, raw_model, optimizer, scheduler, ema,
            )

        if use_ddp_flag:
            dist.barrier()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        save_checkpoint(os.path.join(save_dir, "last.pt"),
                        args.epochs - 1, raw_model, optimizer, scheduler, ema)
        log("Done.")

    if use_ddp_flag:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
