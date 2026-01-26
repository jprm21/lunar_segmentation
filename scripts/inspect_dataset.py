import numpy as np
from collections import Counter
from tqdm import tqdm

from src.datasets.lusnar_dataset import LuSNARDataset

NUM_CLASSES = 5
CLASS_NAMES = {
    0: "Regolith",
    1: "Crater",
    2: "Rock",
    3: "Mountain",
    4: "Sky",
}


def main():
    dataset = LuSNARDataset("data")
    pixel_counter = Counter()

    for _, mask in tqdm(dataset, desc="Scanning masks"):
        mask_np = mask.numpy().flatten()
        pixel_counter.update(mask_np.tolist())

    total_pixels = sum(pixel_counter.values())

    print("\n=== Dataset statistics ===")
    print(f"Total images: {len(dataset)}")
    print(f"Total pixels: {total_pixels}\n")

    for cls in range(NUM_CLASSES):
        count = pixel_counter.get(cls, 0)
        pct = 100.0 * count / total_pixels
        print(f"{cls} ({CLASS_NAMES[cls]:<10}): {count:>12} pixels ({pct:6.2f}%)")


if __name__ == "__main__":
    main()
