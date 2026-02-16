"""
DAM-Net Training Script
=======================
End-to-end training loop with:
* Mixed-precision (AMP)
* Cosine LR schedule with linear warmup
* Gradient clipping
* Checkpointing (best & last)
* TensorBoard logging
* IoU / F1 / Dice validation metrics

Usage
-----
    python -m damnet.train                           # defaults (small model)
    python -m damnet.train --preset tiny --epochs 50
    python -m damnet.train --preset base --batch-size 2 --device cuda:0
    python -m damnet.train --resume G:/Sentinel/OUTPUTS/DAMNet/best.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# AMP compatibility for both CPU and CUDA
try:
    from torch.amp import autocast as _autocast, GradScaler
    def autocast(enabled=True, device_type="cuda"):
        return _autocast(device_type=device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from .config import DAMNetConfig
from .dataset import build_dataloaders
from .losses import BCEDiceLoss
from .model import DAMNet


# ── Utility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, epoch, cfg: DAMNetConfig):
    """Linear warmup + cosine decay."""
    if epoch < cfg.warmup_epochs:
        lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
    else:
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ── Metrics ──────────────────────────────────────────────────────────

class MetricTracker:
    """Running IoU, F1, Dice for binary segmentation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        pred_bin = (pred > threshold).long()
        target_bin = target.long()
        self.tp += ((pred_bin == 1) & (target_bin == 1)).sum().item()
        self.fp += ((pred_bin == 1) & (target_bin == 0)).sum().item()
        self.fn += ((pred_bin == 0) & (target_bin == 1)).sum().item()
        self.tn += ((pred_bin == 0) & (target_bin == 0)).sum().item()

    @property
    def iou(self) -> float:
        return self.tp / max(self.tp + self.fp + self.fn, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-8)

    @property
    def dice(self) -> float:
        return 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / max(total, 1)


# ── Training loop ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg,
                    use_amp=False):
    model.train()
    total_loss = 0.0
    metrics = MetricTracker()

    for batch_idx, batch in enumerate(loader):
        pre = batch["pre_image"].to(device, non_blocking=True)
        post = batch["post_image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp, device_type=device.type):
            out = model(pre, post, return_logits=True)
            logits, prob = out["logits"], out["prob"]
            loss = criterion(logits, label)

        scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        metrics.update(prob.detach(), label)

        if (batch_idx + 1) % 20 == 0:
            print(f"    batch {batch_idx + 1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  IoU={metrics.iou:.4f}")

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, use_amp=False):
    model.eval()
    total_loss = 0.0
    metrics = MetricTracker()

    for batch in loader:
        pre = batch["pre_image"].to(device, non_blocking=True)
        post = batch["post_image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=use_amp, device_type=device.type):
            out = model(pre, post, return_logits=True)
            logits, prob = out["logits"], out["prob"]
            loss = criterion(logits, label)

        total_loss += loss.item()
        metrics.update(prob, label)

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, metrics


# ── Main training function ───────────────────────────────────────────

def train(cfg: DAMNetConfig, device: str = "cuda", resume: str | None = None):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA not available, falling back to CPU")
        device = "cpu"
    device = torch.device(device)
    use_amp = (device.type == "cuda") and cfg.mixed_precision
    print(f"\n{'='*60}")
    print(f"  DAM-Net Training")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Mixed prec   : {use_amp}")
    print(f"  Model config : embed_dims={cfg.embed_dims}, depths={cfg.depths}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Epochs       : {cfg.epochs}")
    print(f"  LR           : {cfg.lr}")
    print(f"  Output       : {cfg.output_dir}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────
    print("Loading datasets …")
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches  : {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────
    model = DAMNet(cfg).to(device)
    param_count = model.count_parameters()
    print(f"  Parameters   : {param_count:,} ({param_count / 1e6:.1f}M)")

    # ── Optimizer & Criterion ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = BCEDiceLoss(cfg.bce_weight, cfg.dice_weight)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    best_iou = 0.0

    # ── Resume ───────────────────────────────────────────────────
    if resume and os.path.isfile(resume):
        print(f"  Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_iou = ckpt.get("best_iou", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best IoU={best_iou:.4f}")

    # ── TensorBoard (optional) ───────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(cfg.output_dir, "logs"))
    except ImportError:
        print("  TensorBoard not installed – skipping logging.")

    # ── Training loop ────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch, cfg)

        print(f"\n── Epoch {epoch + 1}/{cfg.epochs}  (lr={lr:.2e}) ──")

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg,
            use_amp=use_amp)

        val_loss, val_m = validate(model, val_loader, criterion, device, cfg,
                                       use_amp=use_amp)

        elapsed = time.time() - t0

        print(f"  Train  loss={train_loss:.4f}  "
              f"IoU={train_m.iou:.4f}  F1={train_m.f1:.4f}  "
              f"Dice={train_m.dice:.4f}")
        print(f"  Val    loss={val_loss:.4f}  "
              f"IoU={val_m.iou:.4f}  F1={val_m.f1:.4f}  "
              f"Dice={val_m.dice:.4f}  Acc={val_m.accuracy:.4f}")
        print(f"  Time   {elapsed:.1f}s")

        # TensorBoard
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/iou", train_m.iou, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/iou", val_m.iou, epoch)
            writer.add_scalar("val/f1", val_m.f1, epoch)
            writer.add_scalar("val/dice", val_m.dice, epoch)
            writer.add_scalar("lr", lr, epoch)

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": max(best_iou, val_m.iou),
            "config": cfg.__dict__,
        }
        torch.save(ckpt, os.path.join(cfg.output_dir, "last.pt"))

        if val_m.iou > best_iou:
            best_iou = val_m.iou
            torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))
            print(f"  ★ New best IoU = {best_iou:.4f}  (saved)")

    if writer is not None:
        writer.close()

    print(f"\n{'='*60}")
    print(f"  Training complete.  Best val IoU = {best_iou:.4f}")
    print(f"  Checkpoints: {cfg.output_dir}")
    print(f"{'='*60}")
    return model


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DAM-Net flood detector")
    p.add_argument("--preset", choices=["tiny", "small", "base"], default="small",
                   help="Model size preset")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--s1-dir", type=str, default=None,
                   help="Override S1 image directory")
    p.add_argument("--label-dir", type=str, default=None,
                   help="Override label directory")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--train-csv", type=str, default=None)
    p.add_argument("--valid-csv", type=str, default=None)
    p.add_argument("--img-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Build config from preset
    if args.preset == "tiny":
        cfg = DAMNetConfig.tiny()
    elif args.preset == "base":
        cfg = DAMNetConfig.base()
    else:
        cfg = DAMNetConfig.small()

    # Override from CLI
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.s1_dir is not None:
        cfg.s1_dir = args.s1_dir
    if args.label_dir is not None:
        cfg.label_dir = args.label_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.train_csv is not None:
        cfg.train_csv = args.train_csv
    if args.valid_csv is not None:
        cfg.valid_csv = args.valid_csv
    if args.img_size is not None:
        cfg.img_size = args.img_size

    train(cfg, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
