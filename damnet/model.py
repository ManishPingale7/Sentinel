"""
DAM-Net: Full Model
===================
Bitemporal Siamese Vision Transformer with Temporal-Differential Fusion
for pixel-level flood change detection on Sentinel-1 SAR imagery.

Architecture overview
---------------------
1. **PatchEmbed** – overlapping conv embed (stride 4)
2. **SiameseEncoder** – 4 hierarchical stages with weight-shared branches
   * Each stage: TWFE blocks → CTCA → TACE
   * RAM modules compute differential features between branches
   * SemanticToken aggregates global water-body context at stage 4
3. **TDFDecoder** – progressive up-sampling with semantic token injection
4. **SegmentationHead** – 1×1 conv → sigmoid (binary flood mask)

Forward signature::

    mask = model(pre_image, post_image)
    # pre_image, post_image : (B, 2, H, W)  Sentinel-1 VV+VH
    # mask                  : (B, 1, H, W)  flood probability [0, 1]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DAMNetConfig
from .modules import (
    CTCA,
    FFN,
    RAM,
    TACE,
    CrossTemporalAttention,
    DropPath,
    PatchEmbed,
    PatchMerge,
    SemanticToken,
    SpatialReductionAttention,
    TransformerBlock,
)


# =====================================================================
# Siamese Encoder Stage
# =====================================================================

class SiameseStage(nn.Module):
    """One hierarchical stage of the weight-sharing Siamese encoder.

    Contains:
    * ``depth`` TransformerBlocks (TWFE) applied identically to both branches
    * One CTCA module for cross-temporal interaction
    * One TACE module for change enhancement
    """

    def __init__(self, dim: int, depth: int, num_heads: int,
                 sr_ratio: int = 1, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: List[float] | float = 0.0):
        super().__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        # TWFE – self-attention blocks (shared weights for both branches)
        self.twfe_blocks = nn.ModuleList([
            TransformerBlock(
                dim, num_heads, sr_ratio, mlp_ratio, drop, attn_drop, dp)
            for dp in drop_path
        ])

        # CTCA – cross-temporal change attention
        self.ctca = CTCA(dim, num_heads, sr_ratio, attn_drop, drop,
                         drop_path=max(drop_path))

        # TACE – temporal-aware change enhancement
        self.tace = TACE(dim)

        # RAM – residual attention for differential features
        self.ram = RAM(dim)

    def forward(self, feat_pre: torch.Tensor, feat_post: torch.Tensor,
                H: int, W: int):
        """
        Args:
            feat_pre, feat_post : (B, N, C) token sequences per branch
        Returns:
            enhanced_pre, enhanced_post, diff_feat : all (B, N, C)
        """
        # ── TWFE: self-attention on each branch (weight-shared) ──
        for blk in self.twfe_blocks:
            feat_pre = blk(feat_pre, H, W)
            feat_post = blk(feat_post, H, W)

        # ── CTCA: cross-temporal change detection ──
        change_pre, change_post = self.ctca(feat_pre, feat_post, H, W)

        # ── TACE: enhance change signals ──
        enhanced_pre, enhanced_post = self.tace(
            feat_pre, feat_post, change_pre, change_post)

        # ── RAM: differential feature ──
        diff_feat = self.ram(enhanced_pre, enhanced_post)

        return enhanced_pre, enhanced_post, diff_feat


# =====================================================================
# Full Siamese Encoder
# =====================================================================

class SiameseEncoder(nn.Module):
    """4-stage hierarchical Siamese encoder with weight-shared branches."""

    def __init__(self, cfg: DAMNetConfig):
        super().__init__()
        dims = cfg.embed_dims
        depths = cfg.depths
        heads = cfg.num_heads
        srs = cfg.sr_ratios

        # Patch embedding (shared)
        self.patch_embed = PatchEmbed(cfg.in_channels, dims[0])

        # Patch merging between stages (shared)
        self.merges = nn.ModuleList([
            PatchMerge(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

        # Stochastic depth schedule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]
        cur = 0

        # Siamese stages
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = SiameseStage(
                dim=dims[i],
                depth=depths[i],
                num_heads=heads[i],
                sr_ratio=srs[i],
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[cur:cur + depths[i]],
            )
            self.stages.append(stage)
            cur += depths[i]

        # Semantic token at Stage 4
        self.semantic_token = SemanticToken(dims[-1], heads[-1])

        # Final norms
        self.norm_pre = nn.LayerNorm(dims[-1])
        self.norm_post = nn.LayerNorm(dims[-1])

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor):
        """
        Args:
            pre_img, post_img : (B, C_in, H, W)
        Returns:
            multi_pre  : list of (B, N_i, C_i)  per-stage pre features
            multi_post : list of (B, N_i, C_i)  per-stage post features
            multi_diff : list of (B, N_i, C_i)  per-stage differential features
            t_sem      : (B, 1, C_4)            semantic class token
            spatial_sizes : list of (H_i, W_i)
        """
        # Shared patch embedding
        feat_pre, H, W = self.patch_embed(pre_img)
        feat_post, _, _ = self.patch_embed(post_img)

        multi_pre, multi_post, multi_diff = [], [], []
        spatial_sizes = []

        for i, stage in enumerate(self.stages):
            enh_pre, enh_post, diff = stage(feat_pre, feat_post, H, W)

            multi_pre.append(enh_pre)
            multi_post.append(enh_post)
            multi_diff.append(diff)
            spatial_sizes.append((H, W))

            # Down-sample for next stage (except last)
            if i < len(self.stages) - 1:
                H_old, W_old = H, W
                feat_pre, H, W = self.merges[i](enh_pre, H_old, W_old)
                feat_post, _, _ = self.merges[i](enh_post, H_old, W_old)

        # Semantic class token from last-stage features
        combined_last = self.norm_pre(multi_pre[-1]) + self.norm_post(multi_post[-1])
        t_sem = self.semantic_token(combined_last)   # (B, 1, C_4)

        return multi_pre, multi_post, multi_diff, t_sem, spatial_sizes


# =====================================================================
# Temporal-Differential Fusion Decoder (TDF)
# =====================================================================

class TDFBlock(nn.Module):
    """One decoder block: upsample → fuse skip + diff + semantic → refine."""

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int, sem_dim: int):
        super().__init__()
        # Project skip connection (pre+post+diff concat)
        self.skip_proj = nn.Sequential(
            nn.Linear(skip_dim * 3, out_dim),
            nn.GELU(),
        )
        # Project semantic token
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, out_dim),
            nn.GELU(),
        )
        # Fuse up-sampled features + skip + semantic
        total_in = in_dim + out_dim + out_dim  # upsampled + skip + sem
        self.fuse = nn.Sequential(
            nn.Linear(total_in, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

        # Spatial refinement via depth-wise conv to suppress speckle noise
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor, skip_pre: torch.Tensor,
                skip_post: torch.Tensor, skip_diff: torch.Tensor,
                t_sem: torch.Tensor,
                H_up: int, W_up: int, H_skip: int, W_skip: int) -> torch.Tensor:
        """
        Args:
            x         : (B, N_in, C_in)  from previous decoder level
            skip_*    : (B, N_skip, C_skip)  encoder skip connections
            t_sem     : (B, 1, C_sem)
            H_up, W_up, H_skip, W_skip : spatial dims
        Returns:
            out : (B, N_skip, C_out)
        """
        B = x.shape[0]
        C_in = x.shape[-1]

        # Upsample x to match skip spatial size
        x_2d = x.transpose(1, 2).reshape(B, C_in, H_up, W_up)
        x_2d = F.interpolate(x_2d, size=(H_skip, W_skip),
                             mode="bilinear", align_corners=False)
        x_up = x_2d.flatten(2).transpose(1, 2)     # (B, N_skip, C_in)

        # Process skip connections
        skip_cat = torch.cat([skip_pre, skip_post, skip_diff], dim=-1)
        skip_proj = self.skip_proj(skip_cat)         # (B, N_skip, out_dim)

        # Broadcast semantic token to all spatial positions
        N_skip = skip_proj.shape[1]
        sem = self.sem_proj(t_sem).expand(-1, N_skip, -1)  # (B, N_skip, out_dim)

        # Fuse
        fused = torch.cat([x_up, skip_proj, sem], dim=-1)
        out = self.fuse(fused)
        out = self.norm(out)

        # Spatial refinement (2D convs)
        C_out = out.shape[-1]
        out_2d = out.transpose(1, 2).reshape(B, C_out, H_skip, W_skip)
        out_2d = self.refine(out_2d) + out_2d        # residual
        out = out_2d.flatten(2).transpose(1, 2)

        return out


class TDFDecoder(nn.Module):
    """Temporal-Differential Fusion decoder with progressive up-sampling.

    Integrates multi-scale encoder features, differential features, and
    the global class-token to produce a full-resolution feature map.
    """

    def __init__(self, cfg: DAMNetConfig):
        super().__init__()
        dims = cfg.embed_dims       # [C1, C2, C3, C4]
        dec_dim = cfg.decoder_dim
        sem_dim = dims[-1]          # semantic token dim = C4

        num_stages = len(dims)

        # Project stage-4 features to decoder dim
        self.stem = nn.Sequential(
            nn.Linear(dims[-1] * 3, dec_dim),   # pre+post+diff concat
            nn.GELU(),
            nn.Linear(dec_dim, dec_dim),
        )

        # Decoder blocks: from stage 3 → stage 0
        self.blocks = nn.ModuleList()
        for i in range(num_stages - 2, -1, -1):  # stages 2, 1, 0
            self.blocks.append(TDFBlock(
                in_dim=dec_dim,
                skip_dim=dims[i],
                out_dim=dec_dim,
                sem_dim=sem_dim,
            ))

        self.final_norm = nn.LayerNorm(dec_dim)

    def forward(self, multi_pre, multi_post, multi_diff,
                t_sem, spatial_sizes) -> torch.Tensor:
        """
        Returns:
            feat : (B, dec_dim, H_0, W_0)  high-resolution feature map
        """
        num_stages = len(multi_pre)

        # Start from deepest stage
        x = torch.cat([multi_pre[-1], multi_post[-1], multi_diff[-1]], dim=-1)
        x = self.stem(x)                        # (B, N_4, dec_dim)

        H_cur, W_cur = spatial_sizes[-1]

        # Progressive up-sampling through skip levels
        for idx, blk in enumerate(self.blocks):
            level = num_stages - 2 - idx         # 2, 1, 0
            H_skip, W_skip = spatial_sizes[level]
            x = blk(x,
                     multi_pre[level], multi_post[level], multi_diff[level],
                     t_sem,
                     H_cur, W_cur, H_skip, W_skip)
            H_cur, W_cur = H_skip, W_skip

        B, N, C = x.shape
        x = self.final_norm(x)
        x = x.transpose(1, 2).reshape(B, C, H_cur, W_cur)
        return x


# =====================================================================
# Segmentation Head
# =====================================================================

class SegmentationHead(nn.Module):
    """Map decoder features → full-resolution binary mask."""

    def __init__(self, in_dim: int, num_classes: int = 1, upsample: int = 4):
        super().__init__()
        self.upsample = upsample
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, num_classes, 1),
        )

    def forward(self, feat: torch.Tensor, orig_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            feat      : (B, C, H', W') decoder output
            orig_size : (H, W) original input resolution
        Returns:
            mask : (B, num_classes, H, W)
        """
        x = self.head(feat)
        if x.shape[2:] != orig_size:
            x = F.interpolate(x, size=orig_size, mode="bilinear",
                              align_corners=False)
        return x


# =====================================================================
# DAM-Net — Complete Model
# =====================================================================

class DAMNet(nn.Module):
    """Bitemporal Siamese Vision Transformer for SAR flood change detection.

    Inputs
    ------
    pre_img  : (B, C_in, H, W)  – pre-event Sentinel-1 SAR (VV + VH)
    post_img : (B, C_in, H, W)  – post-event Sentinel-1 SAR (VV + VH)

    Output
    ------
    mask : (B, 1, H, W)  – flood probability map ∈ [0, 1]

    When ``single_image=True`` during inference, ``pre_img`` is treated as
    a zero baseline and only ``post_img`` drives the prediction (falls back
    to single-image flood segmentation).
    """

    def __init__(self, cfg: Optional[DAMNetConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = DAMNetConfig.small()
        self.cfg = cfg

        self.encoder = SiameseEncoder(cfg)
        self.decoder = TDFDecoder(cfg)
        self.seg_head = SegmentationHead(cfg.decoder_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # ── Critical: near-zero init the final classifier conv ────────
        # So initial logits ≈ 0  →  sigmoid(0) = 0.5  →  no extreme
        # confidence at start  →  small gradients  →  stable training.
        final_conv = self.seg_head.head[-1]          # last 1×1 conv
        nn.init.normal_(final_conv.weight, std=1e-4)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor,
                return_logits: bool = False) -> torch.Tensor:
        """
        Args:
            pre_img       : (B, C_in, H, W)
            post_img      : (B, C_in, H, W)
            return_logits : if True, return raw logits (for training loss)
        Returns:
            If return_logits=True:  dict with 'logits' and 'prob'
            If return_logits=False: (B, 1, H, W) sigmoid-activated flood probability
        """
        orig_H, orig_W = post_img.shape[2:]

        # Encode both branches
        multi_pre, multi_post, multi_diff, t_sem, sizes = \
            self.encoder(pre_img, post_img)

        # Decode with temporal-differential fusion
        feat = self.decoder(multi_pre, multi_post, multi_diff, t_sem, sizes)

        # Segment
        logits = self.seg_head(feat, (orig_H, orig_W))

        if return_logits:
            return {"logits": logits, "prob": torch.sigmoid(logits)}
        return torch.sigmoid(logits)

    def predict_single(self, post_img: torch.Tensor) -> torch.Tensor:
        """Single-image mode: use zeros as pre-flood baseline."""
        pre_img = torch.zeros_like(post_img)
        return self.forward(pre_img, post_img)

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
