"""
Loss Functions for DAM-Net
==========================
Combined Binary Cross-Entropy + Dice Loss to handle class imbalance
(flood pixels are typically rare relative to land pixels).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    .. math::
        \\mathcal{L}_{Dice} = 1 - \\frac{2\\sum p \\cdot g + \\epsilon}
                                        {\\sum p + \\sum g + \\epsilon}
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, 1, H, W)  predicted probabilities [0, 1]
            target : (B, 1, H, W)  binary ground truth {0, 1}
        """
        pred = pred.flatten(1)
        target = target.flatten(1).float()
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCE and Dice losses.

    Expects **raw logits** (before sigmoid) for the BCE component
    (numerically stable via BCEWithLogitsLoss) and applies sigmoid
    internally for the Dice component.

    .. math::
        \\mathcal{L} = \\alpha \\cdot \\mathcal{L}_{BCE}
                     + \\beta  \\cdot \\mathcal{L}_{Dice}
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth)
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, 1, H, W) raw model output before sigmoid
            target : (B, 1, H, W) binary labels {0, 1}
        """
        target = target.float().clamp(0.0, 1.0)

        # BCE on raw logits (numerically stable — no NaN from sigmoid edge cases)
        bce_loss = self.bce(logits, target)

        # Dice on probabilities
        prob = torch.sigmoid(logits)
        dice_loss = self.dice(prob, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal loss variant for extremely imbalanced scenes.

    .. math::
        FL = -\\alpha (1 - p_t)^{\\gamma} \\log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = target * pred + (1 - target) * (1 - pred)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
