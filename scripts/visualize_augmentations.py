import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.lusnar_dataset import LuSNARDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "artifacts" / "rock_boost_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    0: (187, 70, 156),   # Regolith
    1: (232, 250, 80),   # Rock
    2: (120, 0, 200),    # Crater
    3: (173, 69, 31),    # Mountain
    4: (34, 201, 248),   # Sky
}


def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color

    return color_mask


def save_examples(num_examples=5, sample_idx=0):
    dataset = LuSNARDataset(
        root_dir=DATA_ROOT,
        image_size=256,
        scenes=[1, 2, 4, 6, 8, 9],
        use_class_aware_crop=True,
        crop_size=max(32, (int(0.7 * 256) // 32) * 32),
        target_classes=(1, 2),
        max_crop_tries=15,
        augmentation_profile="rock_boost",
    )

    for i in range(num_examples):
        image, mask = dataset[sample_idx]

        image_np = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        mask_np = mask.numpy()
        color_mask = colorize_mask(mask_np)
        overlay = (0.65 * image_np + 0.35 * color_mask).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("RGB")
        axes[0].imshow(image_np)
        axes[0].axis("off")

        axes[1].set_title("Mask")
        axes[1].imshow(color_mask)
        axes[1].axis("off")

        axes[2].set_title("Overlay")
        axes[2].imshow(overlay)
        axes[2].axis("off")

        fig.tight_layout()
        out_path = OUT_DIR / f"rock_boost_example_{i + 1}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    save_examples(num_examples=5, sample_idx=100)
