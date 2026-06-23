#!/usr/bin/env python3
"""Fix 16-bit grayscale images by converting them to properly scaled 8-bit RGB.

PIL's default .convert("RGB") on 16-bit grayscale images (mode "I;16", range
0-65535) does not rescale to the 8-bit range correctly, producing washed-out or
near-white images. This script detects that mode and rescales explicitly before
converting, using the same logic already validated in scripts/test.py's
load_image_safe(). All other images are passed through unchanged (mode-normalized
to RGB only), so the output folder is a complete, ready-to-use replacement for the
input folder.

Usage:
    python fix_16bit_images.py \
        --input_dir  path/to/real_images \
        --output_dir path/to/real_images_fixed \
        --scale_mode per-image

    # Preview what would happen without writing anything:
    python fix_16bit_images.py \
        --input_dir  path/to/real_images \
        --output_dir path/to/real_images_fixed \
        --dry_run
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert 16-bit grayscale ('I;16') images to properly scaled 8-bit "
            "RGB. Other images are passed through unchanged (mode-normalized only)."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory to scan for images.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where all images (converted or unchanged) are written.",
    )
    parser.add_argument(
        "--scale_mode",
        choices=["per-image", "fixed"],
        default="per-image",
        help=(
            "'per-image' -> scale 16-bit images by their own max value (default). "
            "'fixed' -> scale by the theoretical max of 65535."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input_dir recursively, preserving subfolder structure in output_dir.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would be converted, without writing any output files.",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError(
            "--input_dir and --output_dir must be different paths "
            "to avoid overwriting source data."
        )


def find_images(input_dir, recursive):
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def convert_16bit_to_8bit(img, scale_mode):
    """Rescale a 16-bit grayscale PIL image to 8-bit RGB. Same logic as
    scripts/test.py's load_image_safe()."""
    arr = np.asarray(img).astype(np.float32)

    if scale_mode == "fixed":
        denom = 65535.0
    else:  # "per-image"
        denom = float(arr.max()) if arr.max() > 0 else 1.0

    arr_8bit = np.clip(arr / denom * 255.0, 0, 255).astype(np.uint8)
    rgb_image = Image.fromarray(arr_8bit, mode="L").convert("RGB")

    mean_before = float(arr.mean())
    mean_after = float(arr_8bit.mean())
    return rgb_image, mean_before, mean_after


def main():
    args = parse_args()
    validate_args(args)

    image_paths = find_images(args.input_dir, args.recursive)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    print(f"[INFO] Found {len(image_paths)} images in {args.input_dir}")
    print(f"[INFO] Recursive scan : {args.recursive}")
    print(f"[INFO] Scale mode     : {args.scale_mode}")
    print(f"[INFO] Dry run        : {args.dry_run}")
    if not args.dry_run:
        print(f"[INFO] Output dir     : {args.output_dir}")
    print()

    converted_16bit = []
    passed_through = []
    failed = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        relative_path = image_path.relative_to(args.input_dir)

        try:
            with Image.open(image_path) as img:
                img.load()  # force read now, catch errors here rather than later

                if img.mode == "I;16":
                    result_img, mean_before, mean_after = convert_16bit_to_8bit(
                        img, args.scale_mode
                    )
                    converted_16bit.append(str(relative_path))
                    print(
                        f"  [16-bit] {relative_path} | mode=I;16 | "
                        f"mean_before={mean_before:.1f} -> mean_after={mean_after:.1f}"
                    )
                else:
                    result_img = img.convert("RGB")
                    passed_through.append(str(relative_path))

        except Exception as error:  # noqa: BLE001
            warnings.warn(f"Failed to open {image_path}: {error}", RuntimeWarning)
            failed.append(str(relative_path))
            continue

        if not args.dry_run:
            # Force .png on save to avoid JPEG recompression artifacts
            output_path = (args.output_dir / relative_path).with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(output_path)

    print()
    print("[INFO] Summary")
    print(f"  Total scanned        : {len(image_paths)}")
    print(f"  Converted (16-bit)   : {len(converted_16bit)}")
    print(f"  Passed through       : {len(passed_through)}")
    print(f"  Failed to open        : {len(failed)}")

    if converted_16bit:
        print("\n[INFO] 16-bit images converted:")
        for name in converted_16bit:
            print(f"    - {name}")

    if failed:
        print("\n[WARN] Files that failed to open:")
        for name in failed:
            print(f"    - {name}")

    if args.dry_run:
        print("\n[INFO] Dry run complete — no files were written.")
    else:
        print(f"\n[INFO] All output images saved to: {args.output_dir}")
        print("[INFO] Note: all filenames were saved with .png extension.")


if __name__ == "__main__":
    main()
