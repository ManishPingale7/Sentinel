"""
DAM-Net Training on Google Colab
=================================
Copy-paste this entire file into a Colab notebook cell-by-cell.

Prerequisites:
  Upload these folders to Google Drive under "My Drive/Sentinel/":
    - DATA/S1Hand/           (252+ S1 SAR tiles)
    - DATA/LabelHand/        (252+ label tiles)
    - DATA/flood_train_data.csv
    - DATA/flood_valid_data.csv
    - damnet/                (the entire package folder)
"""

# =====================================================================
# CELL 1: Mount Google Drive
# =====================================================================
from google.colab import drive
drive.mount('/content/drive')

# =====================================================================
# CELL 2: Install dependencies
# =====================================================================
# !pip install rasterio

# =====================================================================
# CELL 3: Setup paths & copy files to Colab local storage (faster I/O)
# =====================================================================
import os, shutil

# === EDIT THIS if your Drive folder is different ===
DRIVE_ROOT = "/content/drive/MyDrive/Sentinel"

# Copy damnet package to Colab
if os.path.exists("/content/damnet"):
    shutil.rmtree("/content/damnet")
shutil.copytree(f"{DRIVE_ROOT}/damnet", "/content/damnet")

# Copy data to local (much faster than reading from Drive during training)
os.makedirs("/content/data/S1Hand", exist_ok=True)
os.makedirs("/content/data/LabelHand", exist_ok=True)

print("Copying S1 tiles...")
for f in os.listdir(f"{DRIVE_ROOT}/DATA/S1Hand"):
    src = f"{DRIVE_ROOT}/DATA/S1Hand/{f}"
    dst = f"/content/data/S1Hand/{f}"
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

print("Copying label tiles...")
for f in os.listdir(f"{DRIVE_ROOT}/DATA/LabelHand"):
    src = f"{DRIVE_ROOT}/DATA/LabelHand/{f}"
    dst = f"/content/data/LabelHand/{f}"
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

# Copy CSVs
shutil.copy2(f"{DRIVE_ROOT}/DATA/flood_train_data.csv", "/content/data/")
shutil.copy2(f"{DRIVE_ROOT}/DATA/flood_valid_data.csv", "/content/data/")

s1_count = len(os.listdir("/content/data/S1Hand"))
lbl_count = len(os.listdir("/content/data/LabelHand"))
print(f"\nReady! S1 tiles: {s1_count}, Labels: {lbl_count}")

# =====================================================================
# CELL 4: Verify GPU
# =====================================================================
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# =====================================================================
# CELL 5: Quick data check
# =====================================================================
import sys
sys.path.insert(0, "/content")

from damnet.dataset import Sen1Floods11Dataset

train_ds = Sen1Floods11Dataset(
    csv_path="/content/data/flood_train_data.csv",
    s1_dir="/content/data/S1Hand",
    label_dir="/content/data/LabelHand",
    img_size=512,
    augment=True,
)
sample = train_ds[0]
print(f"Train samples: {len(train_ds)}")
print(f"post_image: {sample['post_image'].shape}")
print(f"label:      {sample['label'].shape}, unique={sample['label'].unique().tolist()}")
print("Data OK!")

# =====================================================================
# CELL 6: TRAIN  (this is the main cell — takes ~2-4 hours)
# =====================================================================
from damnet.config import DAMNetConfig
from damnet.train import train

cfg = DAMNetConfig.small()          # Use .tiny() for quick test, .small() for real training

# Override paths for Colab
cfg.train_csv = "/content/data/flood_train_data.csv"
cfg.valid_csv = "/content/data/flood_valid_data.csv"
cfg.s1_dir    = "/content/data/S1Hand"
cfg.label_dir = "/content/data/LabelHand"
cfg.output_dir = "/content/damnet_output"

# Training hyperparams (adjust as needed)
cfg.epochs     = 100
cfg.batch_size = 4          # T4=4, A100=8, L4=4
cfg.lr         = 6e-5
cfg.img_size   = 512        # Full resolution

model = train(cfg, device="cuda")

# =====================================================================
# CELL 7: Copy checkpoint back to Google Drive (IMPORTANT!)
# =====================================================================
drive_output = f"{DRIVE_ROOT}/OUTPUTS/DAMNet"
os.makedirs(drive_output, exist_ok=True)

shutil.copy2("/content/damnet_output/best.pt", f"{drive_output}/best.pt")
shutil.copy2("/content/damnet_output/last.pt", f"{drive_output}/last.pt")
print(f"Checkpoints saved to Drive: {drive_output}")

# =====================================================================
# CELL 8: Quick test inference on a sample
# =====================================================================
from damnet.inference import load_damnet, predict_tiled, compute_flood_metrics
from damnet.dataset import _read_tif, _normalize_sar
import numpy as np

model = load_damnet("/content/damnet_output/best.pt", device="cuda")

# Pick a test sample
sample = train_ds[10]
pre = sample["pre_image"].numpy()
post = sample["post_image"].numpy()
label = sample["label"].numpy().squeeze()

mask = predict_tiled(model, pre, post, tile_size=512, device="cuda")
metrics = compute_flood_metrics(mask, pixel_res_m=10.0)

print(f"\nInference result:")
print(f"  Flood pixels:   {metrics['flood_pixels']}")
print(f"  Inundated area: {metrics['inundated_area_km2']:.4f} km²")
print(f"  Flood fraction: {metrics['flood_fraction']*100:.2f}%")

# Compare with ground truth
gt_flood = (label == 1).sum()
pred_flood = mask.sum()
intersection = ((mask == 1) & (label == 1)).sum()
union = ((mask == 1) | (label == 1)).sum()
iou = intersection / max(union, 1)
print(f"\n  Ground truth flood px: {gt_flood}")
print(f"  Predicted flood px:   {pred_flood}")
print(f"  IoU: {iou:.4f}")

# =====================================================================
# CELL 9: Visualize a prediction
# =====================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(post[0], cmap='gray')
axes[0].set_title('Post-flood SAR (VV)')
axes[0].axis('off')

axes[1].imshow(label, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title(f'Ground Truth (flood={int(gt_flood)} px)')
axes[1].axis('off')

axes[2].imshow(post[0], cmap='gray')
overlay = np.zeros((*mask.shape, 4))
overlay[mask == 1] = [1, 0, 0, 0.5]
axes[2].imshow(overlay)
axes[2].set_title(f'DAM-Net Prediction (IoU={iou:.3f})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
