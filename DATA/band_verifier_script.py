import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path to the file
file_path = 'G:/Sentinel/DATA/000028_s2_before_flood--Paraguay_44682_S2Hand.tif'


def verify_and_plot(path):
    with rasterio.open(path) as src:
        print(f"--- Metadata for {path} ---")
        print(f"Band Count: {src.count}")
        print(f"Width/Height: {src.width}x{src.height}")
        print(f"Coordinate System: {src.crs}")

        # Read all bands
        data = src.read()

        # Check scaling (Sentinel-2 L2A is usually scaled by 10,000)
        max_val = np.nanmax(data)
        scaling_warning = ""
        if max_val > 1.1:
            scaling_warning = "⚠️ DATA IS INTEGER-SCALED. Prithvi often expects 0-1 reflectance."

        # Prithvi Band Order: B2(Blue), B3(Green), B4(Red), B8A(Narrow NIR), B11(SWIR1), B12(SWIR2)
        # Standard Indices (1-based): 1, 2, 3, 4, 5, 6

        print(f"\n--- Pixel Stats (Reflectance) ---")
        for i in range(1, src.count + 1):
            band = src.read(i)
            print(
                f"Band {i} | Min: {np.nanmin(band):.4f} | Max: {np.nanmax(band):.4f} | Mean: {np.nanmean(band):.4f}")

        print(f"\n{scaling_warning}")

        if src.count != 6:
            print(
                f"❌ ERROR: Expected 6 bands (Prithvi-style), found {src.count}.")
            return

        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # 1. Natural Color (B4, B3, B2) -> Prithvi indices 3, 2, 1
        # Normalize for display (assuming 10000 scale)
        # Clipping at 0.3 reflectance for contrast
        def norm(b): return np.clip(b / 3000, 0, 1)

        rgb = np.dstack((norm(data[2]), norm(data[1]), norm(data[0])))
        axs[0].imshow(rgb)
        axs[0].set_title("Natural Color (B4, B3, B2)")

        # 2. False Color (B8A, B11, B4) -> Prithvi indices 4, 5, 3
        # Highlights moisture and vegetation
        fcc = np.dstack((norm(data[3]), norm(data[4]), norm(data[2])))
        axs[1].imshow(fcc)
        axs[1].set_title("False Color (B8A, B11, B4)")

        plt.tight_layout()
        plt.show()


verify_and_plot(file_path)
