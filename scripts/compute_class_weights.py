#!/usr/bin/env python3
"""
Compute class weights for LuSNAR semantic segmentation dataset
based on pixel frequency.
"""

import os
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm


LABEL_EXTENSIONS = [".png", ".jpg"]


def is_label_file(filename):
    return any(filename.endswith(ext) for ext in LABEL_EXTENSIONS)


def compute_pixel_counts(dataset_root):
    """
    Traverse dataset and count pixels per class.
    Assumes labels are stored in:
    Moon_x/imageY/label/*.png
    """
    pixel_counter = Counter()
    total_pixels = 0

    for moon in sorted(os.listdir(dataset_root)):
        moon_path = os.path.join(dataset_root, moon)
        if not os.path.isdir(moon_path):
            continue

        for image_dir in os.listdir(moon_path):
            image_path = os.path.join(moon_path, image_dir)
            label_dir = os.path.join(image_path, "label")

            if not os.path.isdir(label_dir):
                continue

            for label_file in os.listdir(label_dir):
                if not is_label_file(label_file):
                    continue

                label_path = os.path.join(label_dir, label_file)
                label = np.array(Image.open(label_path))

                values, counts = np.unique(label, return_counts=True)
                for v, c in zip(values, counts):
                    pixel_counter[int(v)] += int(c)
                    total_pixels += int(c)

    return pixel_counter, total_pixels


def compute_class_weights(pixel_counter, total_pixels):
    num_classes = len(pixel_counter)
    weights = {}

    for cls, count in pixel_counter.items():
        weights[cls] = total_pixels / (num_classes * count)

    return weights


def main():
    dataset_root = "data"

    print("Scanning dataset...")
    pixel_counter, total_pixels = compute_pixel_counts(dataset_root)

    print("\n=== Pixel statistics ===")
    for cls in sorted(pixel_counter):
        pct = 100 * pixel_counter[cls] / total_pixels
        print(f"Class {cls}: {pixel_counter[cls]:>12} pixels ({pct:5.2f}%)")

    weights = compute_class_weights(pixel_counter, total_pixels)

    print("\n=== Suggested class weights ===")
    for cls in sorted(weights):
        print(f"Class {cls}: {weights[cls]:.4f}")

    # Torch-ready tensor format
    weight_list = [weights[i] for i in sorted(weights)]
    print("\nTorch tensor:")
    print(f"torch.tensor({weight_list}, dtype=torch.float32)")


if __name__ == "__main__":
    main()
