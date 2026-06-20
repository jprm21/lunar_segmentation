#!/usr/bin/env python3
"""Stratified train/validation/test split by scene folder.

Splits images independently within each scene folder using the same proportions,
then combines the results into single train/val/test output folders. This ensures
every scene contributes proportionally to each split, avoiding the case where a
small or unusual scene ends up entirely in one split.

Expects each scene folder to follow the LuSNAR-style layout:
    scene_dir/color/*.jpg (or .png)
    scene_dir/label/*.png

Usage:
    python split_by_scene.py \
        --scene_dirs path/to/escena1 path/to/escena2 path/to/escena3 \
        --output_dir path/to/output \
        --train_pct 0.70 --val_pct 0.15 --test_pct 0.15
"""

import argparse
import random
import shutil
import warnings
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split by scene folder."
    )
    parser.add_argument(
        "--scene_dirs",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "List of scene folders, each containing 'color/' and 'label/' "
            "subfolders. Space-separated, e.g. --scene_dirs escena1 escena2 escena3"
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help=(
            "Output root. Creates train/color, train/label, val/color, val/label, "
            "test/color, test/label inside it."
        ),
    )
    parser.add_argument(
        "--train_pct", type=float, default=0.70, help="Train fraction. Default: 0.70"
    )
    parser.add_argument(
        "--val_pct", type=float, default=0.15, help="Validation fraction. Default: 0.15"
    )
    parser.add_argument(
        "--test_pct", type=float, default=0.15, help="Test fraction. Default: 0.15"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility. Default: 42"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of the default symlink (use on filesystems without symlink support).",
    )
    return parser.parse_args()


def validate_args(args):
    for scene_dir in args.scene_dirs:
        if not scene_dir.is_dir():
            raise NotADirectoryError(f"Scene folder not found: {scene_dir}")
        if not (scene_dir / "color").is_dir():
            raise NotADirectoryError(f"Missing 'color' subfolder in: {scene_dir}")
        if not (scene_dir / "label").is_dir():
            raise NotADirectoryError(f"Missing 'label' subfolder in: {scene_dir}")

    total_pct = args.train_pct + args.val_pct + args.test_pct
    if abs(total_pct - 1.0) > 1e-6:
        raise ValueError(
            f"train_pct + val_pct + test_pct must equal 1.0, got {total_pct}"
        )


def find_pairs(scene_dir):
    """Find (image_path, label_path) pairs matched by filename stem."""
    color_dir = scene_dir / "color"
    label_dir = scene_dir / "label"

    pairs = []
    for image_path in sorted(color_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.png"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            warnings.warn(
                f"No matching label for {image_path.name} in {label_dir}", RuntimeWarning
            )
    return pairs


def split_pairs(pairs, train_pct, val_pct, seed):
    """Shuffle and split a list of pairs into train/val/test using given proportions."""
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_pct)
    n_val = round(n * val_pct)
    # Test gets the remainder to avoid rounding drift losing/gaining a sample
    n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test, n_test


def place_file(src_path, dst_path, use_copy):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    if use_copy:
        shutil.copy2(src_path, dst_path)
    else:
        try:
            dst_path.symlink_to(src_path.resolve())
        except OSError:
            # Fallback to copy if symlinks aren't supported on this filesystem
            shutil.copy2(src_path, dst_path)


def main():
    args = parse_args()
    validate_args(args)

    splits = {"train": [], "val": [], "test": []}

    print(f"[INFO] Scenes to process: {len(args.scene_dirs)}")
    print(
        f"[INFO] Split proportions: train={args.train_pct:.0%} "
        f"val={args.val_pct:.0%} test={args.test_pct:.0%}"
    )
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] File mode: {'copy' if args.copy else 'symlink (fallback to copy)'}")
    print()

    for scene_dir in args.scene_dirs:
        pairs = find_pairs(scene_dir)
        if not pairs:
            warnings.warn(f"No valid pairs found in {scene_dir}, skipping.", RuntimeWarning)
            continue

        train, val, test, n_test = split_pairs(pairs, args.train_pct, args.val_pct, args.seed)

        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)

        print(
            f"[INFO] {scene_dir.name:20s} total={len(pairs):4d}  "
            f"train={len(train):4d}  val={len(val):4d}  test={len(test):4d}"
        )

    print()
    print("[INFO] Combined totals:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {len(splits[split_name])} images")

    print()
    print("[INFO] Placing files...")
    for split_name, pairs in splits.items():
        color_out = args.output_dir / split_name / "color"
        label_out = args.output_dir / split_name / "label"
        for image_path, label_path in pairs:
            place_file(image_path, color_out / image_path.name, args.copy)
            place_file(label_path, label_out / label_path.name, args.copy)

    print(f"[INFO] Done. Output written to: {args.output_dir}")

    # Warn if any scene contributed zero images to test (small-scene edge case)
    for scene_dir in args.scene_dirs:
        pairs = find_pairs(scene_dir)
        if pairs:
            _, _, _, n_test = split_pairs(pairs, args.train_pct, args.val_pct, args.seed)
            if n_test == 0:
                warnings.warn(
                    f"Scene '{scene_dir.name}' contributed 0 images to the test split "
                    f"(only {len(pairs)} images total, test_pct={args.test_pct:.0%}). "
                    "Consider a higher test_pct or merging small scenes.",
                    RuntimeWarning,
                )


if __name__ == "__main__":
    main()
