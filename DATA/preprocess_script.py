import rasterio
import numpy as np

input_path = "G:/Sentinel/DATA/000028_s2_before_flood--Paraguay_44682_S2Hand.tif"
output_path = "G:/Sentinel/DATA/000028_prithvi_ready.tif"

with rasterio.open(input_path) as src:
    # 1. Select the correct indices (1-based for rasterio)
    # Prithvi wants: Blue(1), Green(2), Red(3), NIR(4), SWIR1(5), SWIR2(6)
    indices = [1, 2, 3, 4, 5, 6]

    # 2. Read and Scale to 0-1 float32
    data = src.read(indices).astype(np.float32) / 10000.0

    # 3. Prepare Metadata
    meta = src.meta.copy()
    meta.update({
        "count": 6,
        "dtype": "float32",
        "nodata": 0
    })

    # 4. Save the "Clean" 6-band file
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data)

print(f"✅ Created 6-band Prithvi-ready file at: {output_path}")
