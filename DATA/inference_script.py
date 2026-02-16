from mmseg.apis import init_segmentor, inference_segmentor
import mmcv
import matplotlib.pyplot as plt

# 1. Setup paths
config_file = 'G:/Sentinel/hls-foundation-os/configs/sen1floods11_config.py'
checkpoint_file = 'G:/Sentinel/model/sen1floods11_Prithvi_100M.pth'
input_tif = 'India_774689_S2Hand_6band.tif'

# 2. Initialize the model
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 3. Run the "Brain"
# result is a list containing the segmentation mask
result = inference_segmentor(model, input_tif)

# 4. Extract and Save the Mask
# Value 1 = Flood, Value 0 = Land, Value -1 = Clouds
mask = result[0]
mmcv.imwrite(mask, 'flood_detection_mask.png')

# Optional: Quick visualization
plt.imshow(mask, cmap='Blues')
plt.title("AI Detected Flood Zones")
plt.show()
