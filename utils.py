"""
Утилиты: seed control, EMA, logging.
"""

import os
import random
import copy
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


# ──────────────────── Seed Control ─────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────── EMA ──────────────────────────────────────────────

class EMA:
    """Exponential Moving Average для весов модели."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ──────────────────── Cosine Schedule with Warmup ──────────────────────

def cosine_scheduler(optimizer, base_lr: float, warmup_epochs: int,
                     total_epochs: int, min_lr: float = 1e-6):
    """Возвращает лямбда-scheduler: warmup + cosine decay."""
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(0.01, epoch / max(1, warmup_epochs))  # min 1% LR
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────── Logging ──────────────────────────────────────────

def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────── Checkpoint ───────────────────────────────────────

def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optimizer, scheduler, ema: EMA = None,
                    metrics: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None,
                    scheduler=None, ema: EMA = None) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if ema and "ema" in state:
        ema.load_state_dict(state["ema"])
    return state.get("epoch", 0)


# ──────────────────── Param Count ──────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }
