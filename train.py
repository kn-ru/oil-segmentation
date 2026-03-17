"""
Training loop для DualPolMAFUformerMIL.

Улучшения:
  - Deep supervision
  - Boundary loss + OHEM
  - Class weights
  - Gradient accumulation
  - EMA, AMP, cosine+warmup

Использование:
    python train.py
    python train.py --epochs 50 --batch-size 1 --no-amp
"""

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train DualPolMAFUformerMIL")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps")
    return parser.parse_args()


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
):
    model.train()
    running = {"total": 0, "cls": 0, "seg": 0, "ds": 0}
    n_batches = 0
    accum_steps = cfg.train.grad_accum_steps

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        vv = batch["vv"].to(device)
        vh = batch["vh"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type="cuda", enabled=cfg.train.use_amp):
            outputs = model(vv, vh)

            loss_dict = criterion(
                outputs["class_logits"], labels,
                outputs["mask_logits"], masks,
                aux_logits=outputs.get("aux_logits"),
                ds_weights=outputs.get("ds_weights"),
            )

            # Scale loss for gradient accumulation
            loss = loss_dict["total"] / accum_steps

        if cfg.train.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every accum_steps
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if cfg.train.use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

            optimizer.zero_grad()
            ema.update(model)

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k].item()
        n_batches += 1

        if (batch_idx + 1) % cfg.train.log_every == 0:
            avg = {k: v / n_batches for k, v in running.items()}
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={avg['total']:.4f} cls={avg['cls']:.4f} "
                f"seg={avg['seg']:.4f} ds={avg['ds']:.4f}"
            )

    return {k: v / max(1, n_batches) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: TotalLoss,
    device: torch.device,
    cfg: Config,
):
    model.eval()
    metrics = MetricsAccumulator()
    running = {"total": 0, "cls": 0, "seg": 0, "ds": 0}
    n_batches = 0

    for batch in dataloader:
        vv = batch["vv"].to(device)
        vh = batch["vh"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

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
    metric_vals = metrics.compute()
    metric_vals.update({f"loss_{k}": v for k, v in avg_loss.items()})

    return metric_vals, metrics.confusion_matrix()


def main():
    args = parse_args()
    cfg = Config()

    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.no_amp:
        cfg.train.use_amp = False
    if args.backbone:
        cfg.model.backbone_name = args.backbone
    if args.grad_accum:
        cfg.train.grad_accum_steps = args.grad_accum

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(cfg.train.log_dir)

    eff_batch = cfg.data.batch_size * cfg.train.grad_accum_steps
    logger.info(f"Device: {device}")
    logger.info(f"Config: epochs={cfg.train.epochs}, bs={cfg.data.batch_size}, "
                f"grad_accum={cfg.train.grad_accum_steps} (eff_batch={eff_batch}), "
                f"lr={cfg.train.lr}, amp={cfg.train.use_amp}")
    logger.info(f"Loss: seg_w={cfg.train.seg_weight}, ds_w={cfg.train.ds_weight}, "
                f"focal={cfg.train.focal_weight}, dice={cfg.train.dice_weight}, "
                f"boundary={cfg.train.boundary_weight}, ohem={cfg.train.ohem_ratio}")
    logger.info(f"Class weights: {cfg.train.class_weights}")

    # ── Data ──
    logger.info("Building file list...")
    entries = build_file_list(cfg.data.data_dir)
    logger.info(f"Total entries: {len(entries)}")

    train_entries, val_entries = stratified_split(
        entries, cfg.data.val_fraction, cfg.seed)
    logger.info(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    # Normalizer
    norm_path = os.path.join(cfg.train.save_dir, "normalizer.json")
    normalizer = SARNormalizer()
    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        logger.info(f"Loaded normalizer from {norm_path}")
    else:
        logger.info("Fitting normalizer on train data...")
        train_img_paths = [e["image_path"] for e in train_entries]
        normalizer.fit(train_img_paths, cfg.data.clip_percentiles)
        os.makedirs(cfg.train.save_dir, exist_ok=True)
        normalizer.save(norm_path)
        logger.info(f"Saved normalizer to {norm_path}")

    # Oil patch bank для copy-paste аугментации
    logger.info("Building oil patch bank for copy-paste augmentation...")
    oil_train_entries = [e for e in train_entries if e["class_name"] == "oil"]
    oil_patch_bank = build_oil_patch_bank(
        oil_train_entries, normalizer,
        max_images=100, patches_per_image=5,
    )
    n_patches = len(oil_patch_bank["vv"]) if oil_patch_bank else 0
    logger.info(f"  Patch bank: {n_patches} patches from oil images")

    # Augmenter
    augmenter = SARAugmenter(
        speckle_prob=0.5,
        speckle_looks=4,
        radio_prob=0.5,
        radio_shift=0.3,
        radio_scale=0.15,
        gauss_prob=0.3,
        gauss_std=0.1,
        copypaste_prob=0.5,
        copypaste_patches=3,
    )

    # Datasets
    train_ds = OilSpillBagDataset(
        train_entries, normalizer, cfg.data.tile_size,
        force_oil_crop_prob=cfg.data.force_oil_crop_prob,
        force_oil_min_ratio=cfg.data.force_oil_min_ratio,
        augment=True,
        augmenter=augmenter,
        oil_patch_bank=oil_patch_bank,
    )
    val_ds = OilSpillBagDataset(
        val_entries, normalizer, cfg.data.tile_size,
        augment=False,
    )

    train_sampler = ClassBalancedSampler(
        train_entries, cfg.train.lookalike_oversample, cfg.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, sampler=train_sampler,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )

    # ── Model ──
    logger.info("Building model...")
    model = DualPolMAFUformerMIL.from_config(cfg).to(device)
    params = count_parameters(model)
    logger.info(f"Parameters: {params['trainable_M']:.2f}M trainable / "
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = cosine_scheduler(
        optimizer, cfg.train.lr, cfg.train.warmup_epochs, cfg.train.epochs)

    scaler = GradScaler("cuda", enabled=cfg.train.use_amp)
    ema = EMA(model, cfg.train.ema_decay)

    start_epoch = 0
    best_metric = 0

    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, ema)
        logger.info(f"Resumed from epoch {start_epoch}")

    # ── Training Loop ──
    logger.info("Starting training...")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            ema, device, epoch, cfg, logger,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch} done in {elapsed:.0f}s | "
            f"loss={train_losses['total']:.4f} ds={train_losses['ds']:.4f} | "
            f"lr={lr:.2e}"
        )

        # Validation
        if (epoch + 1) % cfg.train.val_every == 0:
            val_metrics, conf_mat = validate(
                ema.shadow, val_loader, criterion, device, cfg)

            logger.info(
                f"Val: macro_f1={val_metrics['macro_f1']:.4f} "
                f"acc={val_metrics['accuracy']:.4f} "
                f"oil_iou={val_metrics['oil_iou']:.4f} "
                f"oil_dice={val_metrics['oil_dice']:.4f} "
                f"loss={val_metrics['loss_total']:.4f}"
            )
            logger.info(f"  Per-class recall: "
                        f"oil={val_metrics['recall_oil']:.3f} "
                        f"look={val_metrics['recall_lookalike']:.3f} "
                        f"nooil={val_metrics['recall_no_oil']:.3f}")
            logger.info(f"  FP area: "
                        f"look={val_metrics['fp_area_lookalike']:.6f} "
                        f"nooil={val_metrics['fp_area_no_oil']:.6f}")

            # Combined score: classification + segmentation
            score = val_metrics["macro_f1"] + val_metrics["oil_dice"]
            if score > best_metric:
                best_metric = score
                save_checkpoint(
                    os.path.join(cfg.train.save_dir, "best.pt"),
                    epoch, model, optimizer, scheduler, ema, val_metrics,
                )
                logger.info(f"  New best! score={score:.4f}")

        if (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.train.save_dir, f"epoch_{epoch}.pt"),
                epoch, model, optimizer, scheduler, ema,
            )

    save_checkpoint(
        os.path.join(cfg.train.save_dir, "last.pt"),
        cfg.train.epochs - 1, model, optimizer, scheduler, ema,
    )
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
