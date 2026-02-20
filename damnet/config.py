"""
DAM-Net Configuration
=====================
All hyper-parameters in one place.  Create presets via class methods.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

_DAMNET_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DAMNetConfig:
    """Complete configuration for DAM-Net architecture + training."""

    # ── Input ────────────────────────────────────────────────────────
    in_channels: int = 2                # VV + VH for Sentinel-1 SAR
    img_size: int = 512                 # Expected H = W of input tiles
    num_classes: int = 1                # Binary flood / no-flood

    # ── Encoder stages ───────────────────────────────────────────────
    embed_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])
    num_heads: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    sr_ratios: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # ── Decoder ──────────────────────────────────────────────────────
    decoder_dim: int = 256              # Unified channel dim inside decoder
    semantic_token_dim: int = 512       # Dim of the global class token

    # ── Training ─────────────────────────────────────────────────────
    lr: float = 6e-5
    weight_decay: float = 0.01
    epochs: int = 100
    batch_size: int = 4
    bce_weight: float = 0.5            # weight of BCE in combined loss
    dice_weight: float = 0.5           # weight of Dice in combined loss
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # ── Data ─────────────────────────────────────────────────────────
    pixel_res_m: float = 10.0          # Sentinel pixel size in metres
    train_csv: str = os.path.join(_DAMNET_BASE, "DATA", "flood_train_data.csv")
    valid_csv: str = os.path.join(_DAMNET_BASE, "DATA", "flood_valid_data.csv")
    test_csv: str = os.path.join(_DAMNET_BASE, "DATA", "flood_test_data.csv")
    s1_dir: str = os.path.join(_DAMNET_BASE, "DATA", "S1Hand")
    label_dir: str = os.path.join(_DAMNET_BASE, "DATA", "LabelHand")
    output_dir: str = os.path.join(_DAMNET_BASE, "OUTPUTS", "DAMNet")

    # ── Misc ─────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True
    mixed_precision: bool = True

    # ── Convenience presets ──────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "DAMNetConfig":
        """Tiny model for quick experiments / limited GPU memory."""
        return cls(
            embed_dims=[32, 64, 160, 256],
            depths=[1, 1, 2, 1],
            num_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1],
            decoder_dim=128,
            semantic_token_dim=256,
            batch_size=8,
        )

    @classmethod
    def small(cls) -> "DAMNetConfig":
        """Small model – good accuracy / speed trade-off."""
        return cls(
            embed_dims=[64, 128, 256, 512],
            depths=[2, 2, 4, 2],
            num_heads=[2, 4, 8, 16],
            sr_ratios=[8, 4, 2, 1],
            decoder_dim=256,
            semantic_token_dim=512,
        )

    @classmethod
    def base(cls) -> "DAMNetConfig":
        """Base model – higher capacity."""
        return cls(
            embed_dims=[64, 128, 320, 512],
            depths=[3, 4, 6, 3],
            num_heads=[2, 4, 8, 16],
            sr_ratios=[8, 4, 2, 1],
            decoder_dim=320,
            semantic_token_dim=512,
            batch_size=2,
        )
