import matplotlib.pyplot as plt
import numpy as np

from src.datasets.lusnar_dataset import LuSNARDataset

CLASS_COLORS = {
    0: (187, 70, 156),   # Regolith
    1: (120, 0, 200),    # Crater
    2: (232, 250, 80),   # Rock
    3: (173, 69, 31),    # Mountain
    4: (34, 201, 248),   # Sky
}

def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color

    return color_mask


dataset = LuSNARDataset("data")
image, mask = dataset[0]

image_np = np.array(image)
mask_np = mask.numpy()
color_mask = colorize_mask(mask_np)

overlay = (0.6 * image_np + 0.4 * color_mask).astype(np.uint8)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("RGB")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(color_mask)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.savefig("overlay_sample.png")
print("Saved overlay_sample.png")
