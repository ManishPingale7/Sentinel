import rasterio
import numpy as np
import os


def slice_for_prithvi(input_path, output_path):
    # indices: Blue, Green, Red, Narrow NIR(8A), SWIR1, SWIR2
    indices = [2, 3, 4, 9, 11, 12]

    with rasterio.open(input_path) as src:
        # Read the 6 specific bands
        data = src.read(indices)

        # Prepare metadata for the new 6-band file
        meta = src.meta.copy()
        meta.update({
            "count": 6,
            "dtype": "float32"
        })

        with rasterio.open(output_path, "w", **meta) as dst:
            # Ensure data is in float32 for model compatibility
            dst.write(data.astype("float32"))

    print(f"✅ Created: {output_path}")


# Example: slice your Bolivia image
slice_for_prithvi("./India_774689_S2Hand.tif",
                  "India_774689_S2Hand_6band.tif")
