from scipy.ndimage import binary_opening
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os



def normalize_band(band):
    band = np.clip(band, 0, np.percentile(band, 98))
    band = (band - band.min()) / (band.max() - band.min())
    return band

# Get all .tif files in the current directory
tif_files = glob.glob("*_S2Hand.tif")
if not tif_files:
    tif_files = glob.glob("*.tif")

for tif_file in tif_files:
    # Skip prediction files if they were picked up
    if "_pred" in tif_file:
        continue
        
    base_name = os.path.splitext(os.path.basename(tif_file))[0]
    pred_path = os.path.join("output_preds", f"{base_name}_pred.tif")
    
    if not os.path.exists(pred_path):
        print(f"Skipping {tif_file}: Prediction file {pred_path} not found.")
        continue

    print(f"Processing {tif_file}...")

    try:
        # Load original RGB
        src = rasterio.open(tif_file)

        red = src.read(4) / 10000
        green = src.read(3) / 10000
        blue = src.read(2) / 10000

        red = normalize_band(red)
        green = normalize_band(green)
        blue = normalize_band(blue)

        rgb_img = np.stack([red, green, blue], axis=-1)

        # Load prediction
        pred_src = rasterio.open(pred_path)
        pred = pred_src.read(1)
        flood_pixels = np.sum(pred == 1)
        area_m2 = flood_pixels * 100  # 10m x 10m pixel
        area_hectares = area_m2 / 10000

        clean_mask = binary_opening(pred == 1, structure=np.ones((3, 3)))

        print(f"Flood Area (hectares) for {base_name}:", area_hectares)

        overlay = rgb_img.copy()
        # Create blue mask
        blue_overlay = np.zeros_like(rgb_img)
        blue_overlay[..., 2] = 1  # full blue channel

        alpha = 0.6  # transparency
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title(f"Original: {base_name}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(rgb_img)
        plt.imshow(blue_overlay, alpha=alpha * (clean_mask))
        plt.title(f"Flood Detection (Prithvi Model)\n{base_name}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        
        src.close()
        pred_src.close()
        
    except Exception as e:
        print(f"Error processing {tif_file}: {e}")
