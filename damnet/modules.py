"""
DAM-Net Sub-Modules
===================
Building blocks for the Siamese Vision Transformer:

* PatchEmbed          – overlapping convolutional patch embedding
* PatchMerge          – 2× spatial down-sampling between stages
* SpatialReductionAttention (SRA) – efficient self-attention with SR
* CrossTemporalAttention (CTA)    – cross-attention between branches
* FFN                 – feed-forward with depth-wise conv
* TransformerBlock    – LN → SRA → LN → FFN  (one self-attention block)
* CTCA                – Cross-Temporal Change Attention wrapper
* TACE                – Temporal-Aware Change Enhancement
* SemanticToken       – learnable class token with cross-attention
"""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Helpers
# =====================================================================

def _pair(x):
    return (x, x) if isinstance(x, int) else x


class DropPath(nn.Module):
    """Stochastic depth (drop entire residual branch)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x.div(keep) * mask


# =====================================================================
# Patch Embedding / Merging
# =====================================================================

class PatchEmbed(nn.Module):
    """Overlapping patch embedding (stride-4, kernel-7)."""

    def __init__(self, in_ch: int = 2, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=7, stride=4, padding=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, C, H, W)
        x = self.proj(x)                       # (B, D, H/4, W/4)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N, D)  N=H*W
        x = self.norm(x)
        return x, H, W


class PatchMerge(nn.Module):
    """Down-sample spatial resolution by 2× between encoder stages."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)                       # (B, out_dim, H/2, W/2)
        B, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N', out_dim)
        x = self.norm(x)
        return x, H2, W2


# =====================================================================
# Attention Variants
# =====================================================================

class SpatialReductionAttention(nn.Module):
    """Multi-head self-attention with spatial reduction (PVT-style).

    When ``sr_ratio > 1`` the key/value spatial dimensions are reduced
    with a strided convolution to keep memory tractable.
    """

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 1,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            kv_in = x.transpose(1, 2).reshape(B, C, H, W)
            kv_in = self.sr(kv_in).flatten(2).transpose(1, 2)  # (B, N', C)
            kv_in = self.sr_norm(kv_in)
        else:
            kv_in = x

        kv = self.kv(kv_in).reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)         # (2, B, heads, N', hd)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossTemporalAttention(nn.Module):
    """Cross-attention: queries from branch A, keys/values from branch B.

    Highlights temporal differences in backscatter between pre/post images.
    Uses spatial reduction on the source (KV) branch for efficiency.
    """

    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 1,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, source: torch.Tensor,
                H: int, W: int) -> torch.Tensor:
        """
        Args:
            query:  (B, N, C) – branch that asks "what changed?"
            source: (B, N, C) – the other temporal branch
        """
        B, N, C = query.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            src2d = source.transpose(1, 2).reshape(B, C, H, W)
            src2d = self.sr(src2d).flatten(2).transpose(1, 2)
            src2d = self.sr_norm(src2d)
        else:
            src2d = source

        kv = self.kv_proj(src2d).reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# =====================================================================
# FFN
# =====================================================================

class FFN(nn.Module):
    """Feed-forward network with depth-wise 3×3 conv (spatial mixing)."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        # spatial mixing via depth-wise conv
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dw_conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =====================================================================
# Transformer Block  (TWFE – Temporal-Wise Feature Extraction)
# =====================================================================

class TransformerBlock(nn.Module):
    """Standard Vision-Transformer block with spatial-reduction attention.

    Acts as the **Temporal-Wise Feature Extraction (TWFE)** module when
    used independently on each temporal branch.
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(
            dim, num_heads, sr_ratio, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


# =====================================================================
# CTCA – Cross-Temporal Change Attention
# =====================================================================

class CTCA(nn.Module):
    """Bidirectional cross-temporal change attention.

    Pre-branch queries the Post-branch (and vice-versa) to detect
    backscatter differences indicative of flooding.
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1,
                 attn_drop: float = 0.0, drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm_q_pre = nn.LayerNorm(dim)
        self.norm_s_post = nn.LayerNorm(dim)
        self.cross_pre2post = CrossTemporalAttention(
            dim, num_heads, sr_ratio, attn_drop, drop)

        self.norm_q_post = nn.LayerNorm(dim)
        self.norm_s_pre = nn.LayerNorm(dim)
        self.cross_post2pre = CrossTemporalAttention(
            dim, num_heads, sr_ratio, attn_drop, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, feat_pre: torch.Tensor, feat_post: torch.Tensor,
                H: int, W: int):
        """
        Returns:
            change_pre  – change signal from perspective of pre-branch
            change_post – change signal from perspective of post-branch
        """
        # Pre queries Post → "what appeared that wasn't here before?"
        change_pre = self.drop_path(
            self.cross_pre2post(
                self.norm_q_pre(feat_pre),
                self.norm_s_post(feat_post), H, W))

        # Post queries Pre → "what disappeared / is different now?"
        change_post = self.drop_path(
            self.cross_post2pre(
                self.norm_q_post(feat_post),
                self.norm_s_pre(feat_pre), H, W))

        return change_pre, change_post


# =====================================================================
# TACE – Temporal-Aware Change Enhancement
# =====================================================================

class TACE(nn.Module):
    """Fuse cross-temporal change signals back into each branch.

    Mechanism:
        enhanced = feat + gate * change
    where ``gate`` is a learned sigmoid gating that suppresses SAR
    speckle noise while preserving genuine flood boundaries.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate_pre = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.gate_post = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm_pre = nn.LayerNorm(dim)
        self.norm_post = nn.LayerNorm(dim)

    def forward(self, feat_pre: torch.Tensor, feat_post: torch.Tensor,
                change_pre: torch.Tensor, change_post: torch.Tensor):
        """
        Args:
            feat_pre/post  : (B, N, C) spatial features per branch
            change_pre/post: (B, N, C) cross-temporal change signals
        Returns:
            enhanced_pre, enhanced_post
        """
        g_pre = self.gate_pre(torch.cat([feat_pre, change_pre], dim=-1))
        enhanced_pre = self.norm_pre(feat_pre + g_pre * change_pre)

        g_post = self.gate_post(torch.cat([feat_post, change_post], dim=-1))
        enhanced_post = self.norm_post(feat_post + g_post * change_post)

        return enhanced_pre, enhanced_post


# =====================================================================
# Semantic / Class Token
# =====================================================================

class SemanticToken(nn.Module):
    """Learnable class token that aggregates global water-body context.

    Appended to Stage-4 features and refined via cross-attention to the
    spatial tokens.  Provides the decoder with global semantic cues.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, N, C)  stage-4 spatial tokens
        Returns:
            t_sem: (B, 1, C)  global semantic token
        """
        B = feats.shape[0]
        t = self.token.expand(B, -1, -1)       # (B, 1, C)
        t = self.norm1(t)
        t_attn, _ = self.cross_attn(t, feats, feats)
        t = t + t_attn
        t = t + self.ffn(self.norm2(t))
        return t


# =====================================================================
# Residual Attention Module (RAM) – differential feature fusion
# =====================================================================

class RAM(nn.Module):
    """Residual Attention Module for inter-branch feature interaction.

    Computes differential features between encoder stages and refines
    them with channel attention.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 16)
        # Channel attention on differential features
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, feat_pre: torch.Tensor, feat_post: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            diff_feat: (B, N, C) – refined temporal-differential features
        """
        diff = feat_post - feat_pre              # element-wise subtraction
        B, N, C = diff.shape
        # channel attention
        w = self.attn(diff.transpose(1, 2))      # (B, C)
        w = w.unsqueeze(1)                        # (B, 1, C)
        diff = diff * w
        diff = self.proj(self.norm(diff))
        return diff
