import torch
import skimage.io as sio
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import numpy as np
from train import get_model_instance_segmentation

device = torch.device('cuda')

# 1. Load Model
model = get_model_instance_segmentation(5)
model.load_state_dict(torch.load("outputs/run1/best_model.pth", map_location=device))
model.to(device)
model.eval()

# 2. Load one test image (Change path to a real test image)
img_path = "data/test_release/c8cb7626-7423-4c1e-a81c-5ff25ea180b3.tif" 
img_array = sio.imread(img_path)
if len(img_array.shape) == 2:
    img_array = np.stack((img_array,)*3, axis=-1)
elif img_array.shape[-1] > 3:
    img_array = img_array[:, :, :3]

# NOTE: Divide by 65535.0 here if you discovered they are 16-bit!
img_tensor = torch.as_tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

# 3. Predict
with torch.no_grad():
    with torch.autocast(device_type='cuda'):
        pred = model([img_tensor.to(device)])[0]

# 4. Draw
# Convert image tensor back to uint8 for drawing
draw_img = (img_tensor * 255).to(torch.uint8)
masks = pred['masks'] > 0.5

if len(masks) > 0:
    # Squeeze the channel dimension out of the masks: [N, 1, H, W] -> [N, H, W]
    masks = masks.squeeze(1)
    res_img = draw_segmentation_masks(draw_img, masks, alpha=0.5, colors="red")
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(F.to_pil_image(res_img))
    plt.axis('off')
    plt.savefig("sanity_check.png", bbox_inches='tight')
    print("Saved sanity_check.png! Open it and look at the masks.")
else:
    print("Model predicted absolutely nothing (0 boxes).")