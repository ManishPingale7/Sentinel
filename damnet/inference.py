"""
DAM-Net Inference & Rapid Disaster Assessment
==============================================
End-to-end inference on Sentinel-1 SAR bitemporal pairs:

1. Load trained DAM-Net checkpoint
2. Run forward pass on pre/post image pairs
3. Produce binary flood mask
4. Compute Trustworthy Actionable Metrics:
   - Total Inundated Area (m² and km²)
   - Change pixels / percentage
   - IoU / F1 vs. ground truth (if available)
5. Save GeoTIFF mask + overlay PNG

Usage
-----
    python -m damnet.inference --checkpoint best.pt \\
        --pre pre_flood.tif --post post_flood.tif --output mask.tif

    python -m damnet.inference --checkpoint best.pt \\
        --post single_image.tif --output mask.tif   # single-image mode
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from .config import DAMNetConfig
from .model import DAMNet


# =====================================================================
# Model Loading
# =====================================================================

def load_damnet(checkpoint_path: str, device: str = "cuda",
                config_override: DAMNetConfig | None = None) -> DAMNet:
    """Load a trained DAM-Net from a checkpoint file.

    The checkpoint stores the config used during training, so the model
    architecture is reconstructed automatically.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild config
    if config_override is not None:
        cfg = config_override
    elif "config" in ckpt:
        cfg = DAMNetConfig(**{k: v for k, v in ckpt["config"].items()
                              if k in DAMNetConfig.__dataclass_fields__})
    else:
        cfg = DAMNetConfig.small()

    model = DAMNet(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    iou = ckpt.get("best_iou", "?")
    print(f"[DAM-Net] Loaded checkpoint: epoch={epoch}, best_iou={iou}")
    print(f"[DAM-Net] Parameters: {model.count_parameters():,}")
    return model


# =====================================================================
# Tile-Based Inference (handles images larger than training crop)
# =====================================================================

@torch.no_grad()
def predict_tiled(
    model: DAMNet,
    pre_img: np.ndarray,
    post_img: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    device: str = "cuda",
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference on (potentially large) SAR images using tiled prediction.

    Args:
        pre_img, post_img : (C, H, W)  float32, already normalized
        tile_size          : crop size matching training resolution
        overlap            : overlap between tiles for smooth merging
        device             : torch device
        threshold          : binarization threshold

    Returns:
        mask : (H, W) uint8  binary flood mask {0, 1}
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    C, H, W = post_img.shape
    stride = tile_size - overlap

    # Pad to multiples of stride
    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride
    pre_padded = np.pad(pre_img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    post_padded = np.pad(post_img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    _, Hp, Wp = post_padded.shape
    prob_map = np.zeros((Hp, Wp), dtype=np.float64)
    count_map = np.zeros((Hp, Wp), dtype=np.float64)

    for y in range(0, Hp - tile_size + 1, stride):
        for x in range(0, Wp - tile_size + 1, stride):
            pre_tile = pre_padded[:, y:y + tile_size, x:x + tile_size]
            post_tile = post_padded[:, y:y + tile_size, x:x + tile_size]

            pre_t = torch.from_numpy(pre_tile).unsqueeze(0).to(device)
            post_t = torch.from_numpy(post_tile).unsqueeze(0).to(device)

            pred = model(pre_t, post_t)             # (1, 1, H, W)
            pred_np = pred.squeeze().cpu().numpy()   # (H, W)

            prob_map[y:y + tile_size, x:x + tile_size] += pred_np
            count_map[y:y + tile_size, x:x + tile_size] += 1.0

    # Average overlapping regions
    count_map = np.maximum(count_map, 1.0)
    prob_map /= count_map

    # Crop back to original size
    prob_map = prob_map[:H, :W]
    mask = (prob_map > threshold).astype(np.uint8)
    return mask


# =====================================================================
# Flood Area Quantification
# =====================================================================

def compute_flood_metrics(
    mask: np.ndarray,
    pixel_res_m: float = 10.0,
    pre_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute trustworthy flood metrics from a binary mask.

    Args:
        mask         : (H, W) uint8 flood mask from DAM-Net
        pixel_res_m  : ground sampling distance in metres
        pre_mask     : optional pre-event mask for change analysis

    Returns:
        Dict with area and change metrics.
    """
    pixel_area_m2 = pixel_res_m ** 2
    pixel_area_km2 = pixel_area_m2 / 1e6

    flood_px = int(mask.sum())
    total_px = mask.size
    flood_m2 = flood_px * pixel_area_m2
    flood_km2 = flood_px * pixel_area_km2

    metrics = {
        "flood_pixels": flood_px,
        "total_pixels": total_px,
        "flood_fraction": flood_px / max(total_px, 1),
        "inundated_area_m2": round(flood_m2, 2),
        "inundated_area_km2": round(flood_km2, 6),
    }

    if pre_mask is not None:
        pre_flood_px = int(pre_mask.sum())
        new_flood = (mask == 1) & (pre_mask == 0)
        receded = (mask == 0) & (pre_mask == 1)
        new_px = int(new_flood.sum())
        receded_px = int(receded.sum())

        metrics.update({
            "pre_flood_pixels": pre_flood_px,
            "pre_flood_km2": round(pre_flood_px * pixel_area_km2, 6),
            "new_flood_pixels": new_px,
            "new_flood_km2": round(new_px * pixel_area_km2, 6),
            "receded_pixels": receded_px,
            "receded_km2": round(receded_px * pixel_area_km2, 6),
            "change_pct": round(
                (new_px / max(pre_flood_px, 1)) * 100, 2),
        })

    return metrics


# =====================================================================
# GeoTIFF Mask Writer
# =====================================================================

def save_mask_geotiff(
    mask: np.ndarray,
    output_path: str,
    reference_tif: str | None = None,
):
    """Save binary mask as a GeoTIFF, copying CRS/transform from reference."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if reference_tif is not None:
        with rasterio.open(reference_tif) as ref:
            profile = ref.profile.copy()
    else:
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": mask.shape[1],
            "height": mask.shape[0],
        }

    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw",
        nodata=255,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

    print(f"  Mask saved → {output_path}")


# =====================================================================
# Overlay Visualization
# =====================================================================

def generate_overlay(
    post_img: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    metrics: Dict[str, float] | None = None,
    title: str = "DAM-Net Flood Detection",
):
    """Generate an overlay PNG with flood regions highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = mask.shape

    # Compose VV band as grayscale background
    if post_img.ndim == 3:
        bg = post_img[0]  # VV band
    else:
        bg = post_img
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: raw SAR
    axes[0].imshow(bg, cmap="gray")
    axes[0].set_title("Post-event SAR (VV)", fontsize=12)
    axes[0].axis("off")

    # Right: overlay
    axes[1].imshow(bg, cmap="gray")
    flood_overlay = np.zeros((H, W, 4), dtype=np.float32)
    flood_overlay[mask == 1] = [1.0, 0.0, 0.0, 0.55]  # red flood
    axes[1].imshow(flood_overlay)

    # Annotate with metrics
    if metrics:
        area_km2 = metrics.get("inundated_area_km2", 0)
        frac = metrics.get("flood_fraction", 0)
        subtitle = f"Inundated: {area_km2:.4f} km² ({frac*100:.1f}% of tile)"
        if "new_flood_km2" in metrics:
            subtitle += f"\nNew flooding: {metrics['new_flood_km2']:.4f} km²"
    else:
        subtitle = ""

    axes[1].set_title(f"{title}\n{subtitle}", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overlay saved → {output_path}")


# =====================================================================
# High-Level Inference Function
# =====================================================================

def run_inference(
    checkpoint_path: str,
    post_path: str,
    pre_path: str | None = None,
    output_dir: str = ".",
    device: str = "cuda",
    pixel_res_m: float = 10.0,
    tile_size: int = 512,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Complete inference pipeline on one bitemporal pair.

    1. Load model
    2. Read SAR GeoTIFFs
    3. Predict flood mask
    4. Compute metrics
    5. Save mask + overlay

    Returns:
        metrics dict
    """
    from .dataset import _read_tif, _normalize_sar

    # Load model
    model = load_damnet(checkpoint_path, device)

    # Load images
    post_img = _read_tif(post_path)
    if post_img.shape[0] > 2:
        post_img = post_img[:2]
    post_img = _normalize_sar(post_img)

    if pre_path is not None:
        pre_img = _read_tif(pre_path)
        if pre_img.shape[0] > 2:
            pre_img = pre_img[:2]
        pre_img = _normalize_sar(pre_img)
    else:
        pre_img = np.zeros_like(post_img)

    # Predict
    print(f"\nPredicting flood mask …")
    t0 = time.time()
    mask = predict_tiled(model, pre_img, post_img,
                         tile_size=tile_size, device=device,
                         threshold=threshold)
    elapsed = time.time() - t0
    print(f"  Prediction done in {elapsed:.1f}s")

    # Metrics
    metrics = compute_flood_metrics(mask, pixel_res_m)
    print(f"\n  ── Flood Assessment ──")
    print(f"  Inundated area : {metrics['inundated_area_km2']:.4f} km²  "
          f"({metrics['inundated_area_m2']:.0f} m²)")
    print(f"  Flood fraction : {metrics['flood_fraction']*100:.2f}%")

    # Save outputs
    stem = os.path.splitext(os.path.basename(post_path))[0]
    mask_path = os.path.join(output_dir, f"{stem}_damnet_mask.tif")
    overlay_path = os.path.join(output_dir, f"{stem}_damnet_overlay.png")

    save_mask_geotiff(mask, mask_path, reference_tif=post_path)
    generate_overlay(post_img, mask, overlay_path, metrics)

    return metrics


# =====================================================================
# Batch Inference for CEMS Folders
# =====================================================================

def run_batch_inference(
    checkpoint_path: str,
    pairs: List[Tuple[str, str]],
    output_dir: str,
    device: str = "cuda",
    pixel_res_m: float = 10.0,
    tile_size: int = 512,
    threshold: float = 0.5,
) -> List[Dict]:
    """Run inference on a list of (pre_path, post_path) pairs.

    Returns list of per-pair metrics dicts.
    """
    from .dataset import _read_tif, _normalize_sar

    model = load_damnet(checkpoint_path, device)
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    for i, (pre_path, post_path) in enumerate(pairs):
        print(f"\n── Pair {i + 1}/{len(pairs)} ──")
        print(f"  Pre  : {pre_path}")
        print(f"  Post : {post_path}")

        post_img = _read_tif(post_path)
        if post_img.shape[0] > 2:
            post_img = post_img[:2]
        post_img = _normalize_sar(post_img)

        pre_img = _read_tif(pre_path)
        if pre_img.shape[0] > 2:
            pre_img = pre_img[:2]
        pre_img = _normalize_sar(pre_img)

        mask = predict_tiled(model, pre_img, post_img,
                             tile_size=tile_size, device=device,
                             threshold=threshold)

        metrics = compute_flood_metrics(mask, pixel_res_m)
        metrics["pre_path"] = pre_path
        metrics["post_path"] = post_path

        stem = os.path.splitext(os.path.basename(post_path))[0]
        mask_path = os.path.join(output_dir, f"{stem}_damnet_mask.tif")
        overlay_path = os.path.join(output_dir, f"{stem}_damnet_overlay.png")

        save_mask_geotiff(mask, mask_path, reference_tif=post_path)
        generate_overlay(post_img, mask, overlay_path, metrics)

        all_metrics.append(metrics)
        print(f"  Inundated: {metrics['inundated_area_km2']:.4f} km²")

    return all_metrics


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DAM-Net Flood Inference")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--post", required=True, help="Post-flood SAR GeoTIFF")
    p.add_argument("--pre", default=None, help="Pre-flood SAR GeoTIFF (optional)")
    p.add_argument("--output", default=r"G:\Sentinel\OUTPUTS\DAMNet",
                   help="Output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pixel-res", type=float, default=10.0,
                   help="Pixel resolution in metres")
    return p.parse_args()


def main():
    args = parse_args()
    run_inference(
        checkpoint_path=args.checkpoint,
        post_path=args.post,
        pre_path=args.pre,
        output_dir=args.output,
        device=args.device,
        tile_size=args.tile_size,
        threshold=args.threshold,
        pixel_res_m=args.pixel_res,
    )


if __name__ == "__main__":
    main()
