import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Path to your SenForFlood image
file_path = "./000028_s2_during_flood--India_631692_S2Hand.tif"

with rasterio.open(file_path) as src:
    print(f"Bands detected in file: {src.count}")

    # In SenForFlood S2 tiles, bands are usually: B2, B3, B4, B8, B11, B12
    # This means Green (B3) is index 2 and NIR (B8) is index 4
    try:
        green = src.read(2).astype(np.float32) / \
            10000  # B3 is usually 2nd band
        nir = src.read(4).astype(np.float32) / \
            10000    # B8/B8A is usually 4th band
    except IndexError:
        print("❌ Error: Band index out of range. Trying standard 13-band indexing...")
        green = src.read(3).astype(np.float32) / 10000
        nir = src.read(8).astype(np.float32) / 10000

# Compute NDWI with a small epsilon to avoid division by zero
ndwi = (green - nir) / (green + nir + 1e-6)

print("NDWI min:", np.nanmin(ndwi))
print("NDWI max:", np.nanmax(ndwi))

# Thresholding: 0.1 is standard for water
water_mask = ndwi > 0.1

# --- Visualization ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title(f"NDWI (Bands 2 & 4)\nMax: {np.nanmax(ndwi):.2f}")
plt.imshow(ndwi, cmap="RdYlBu", vmin=-1, vmax=1)
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Detected Water (Threshold > 0.1)")
plt.imshow(water_mask, cmap="Blues")

plt.subplot(1, 3, 3)
plt.title("Green Band (B3)\nSurface Reflectance")
plt.imshow(green, cmap="gray", vmin=0, vmax=0.3)

plt.tight_layout()
plt.show()
