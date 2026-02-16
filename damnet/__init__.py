"""
DAM-Net: Bitemporal Siamese Vision Transformer for SAR Flood Change Detection
==============================================================================
Custom implementation of a DAM-Net-style architecture for pixel-level flood
segmentation from Sentinel-1 SAR bitemporal pairs.

Modules
-------
- model.DAMNet          : Full model (encoder + decoder + head)
- dataset               : Sen1Floods11 & bitemporal data loaders
- losses                : BCE + Dice combined loss
- train                 : Training loop & CLI
- inference             : Single-pair and batch inference utilities
"""

from .model import DAMNet
from .config import DAMNetConfig

__all__ = ["DAMNet", "DAMNetConfig"]
