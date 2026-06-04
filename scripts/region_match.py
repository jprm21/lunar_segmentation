#!/usr/bin/env python3
"""Region-aware histogram matching for real lunar images.

This preprocessing script adjusts real lunar images using LuSNAR statistics without
running segmentation inference. Dark pixels in the upper image region are treated
as sky and matched to LuSNAR sky statistics; all remaining pixels are treated as
terrain and matched to combined LuSNAR terrain statistics.
"""

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LUSNAR_RGB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKY_CLASS_ID = 4
CLASS_COLORS = {
    (187, 70, 156): 0,  # Lunar regolith
    (120, 0, 200): 1,  # Impact crater
    (232, 250, 80): 2,  # Rock
    (173, 69, 31): 3,  # Mountain
    (34, 201, 248): 4,  # Sky
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess real lunar images with separate sky and terrain "
            "histogram matching against LuSNAR reference statistics."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing real lunar images to process.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where adjusted images and comparison images will be saved.",
    )
    parser.add_argument(
        "--lusnar_dir",
        required=True,
        type=Path,
        help=(
            "LuSNAR data root. The script looks for scene_XX image folders with "
            "matching masks, and also supports the repo's Moon_*/image0/color + "
            "label layout."
        ),
    )
    parser.add_argument(
        "--lusnar_samples",
        type=int,
        default=300,
        help="Number of LuSNAR image pairs to sample. Default: 300.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=384,
        help="Resize all images to this square size. Default: 384.",
    )
    parser.add_argument(
        "--sky_threshold",
        type=float,
        default=40,
        help=(
            "Grayscale intensity (0-255) below which a pixel in the upper real "
            "image is considered sky. Default: 40."
        ),
    )
    parser.add_argument(
        "--upper_fraction",
        type=float,
        default=0.55,
        help=(
            "Fraction of image height from top where sky pixels are searched. "
            "Default: 0.55."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input image directory not found: {args.input_dir}")
    if not args.lusnar_dir.is_dir():
        raise NotADirectoryError(f"LuSNAR directory not found: {args.lusnar_dir}")
    if args.lusnar_samples <= 0:
        raise ValueError("--lusnar_samples must be a positive integer")
    if args.image_size <= 0:
        raise ValueError("--image_size must be a positive integer")
    if not 0 <= args.sky_threshold <= 255:
        raise ValueError("--sky_threshold must be between 0 and 255")
    if not 0 < args.upper_fraction <= 1:
        raise ValueError("--upper_fraction must be in the range (0, 1]")


def find_input_images(input_dir):
    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def collect_scene_pairs(lusnar_dir, lusnar_masks_dir):
    pairs = []
    scene_dirs = [path for path in sorted(lusnar_dir.glob("scene_*")) if path.is_dir()]

    for scene_dir in scene_dirs:
        mask_scene_dir = lusnar_masks_dir / scene_dir.name
        for image_path in sorted(scene_dir.iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in LUSNAR_RGB_EXTENSIONS
            ):
                continue

            mask_path = mask_scene_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                pairs.append((image_path, mask_path))
            else:
                warnings.warn(
                    f"Skipping LuSNAR image without matching mask: "
                    f"{image_path} -> {mask_path}",
                    RuntimeWarning,
                )

    return pairs


def collect_moon_pairs(lusnar_dir):
    pairs = []
    for moon_dir in sorted(lusnar_dir.glob("Moon_*")):
        if not moon_dir.is_dir():
            continue
        for image_dir in sorted(moon_dir.iterdir()):
            rgb_dir = image_dir / "color"
            mask_dir = image_dir / "label"
            if not rgb_dir.is_dir() or not mask_dir.is_dir():
                continue
            for image_path in sorted(rgb_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in LUSNAR_RGB_EXTENSIONS
                ):
                    continue
                mask_path = mask_dir / f"{image_path.stem}.png"
                if mask_path.exists():
                    pairs.append((image_path, mask_path))
                else:
                    warnings.warn(
                        f"Skipping LuSNAR image without matching mask: "
                        f"{image_path} -> {mask_path}",
                        RuntimeWarning,
                    )
    return pairs


def collect_lusnar_pairs(lusnar_dir, lusnar_masks_dir):
    pairs = collect_scene_pairs(lusnar_dir, lusnar_masks_dir)
    if pairs:
        return pairs
    return collect_moon_pairs(lusnar_dir)


def mask_to_class_ids(mask_pil):
    mask_np = np.asarray(mask_pil)

    if mask_np.ndim == 2:
        return mask_np.astype(np.int64)

    if mask_np.ndim == 3 and mask_np.shape[-1] >= 3:
        rgb_mask = mask_np[..., :3]
        class_mask = np.zeros(rgb_mask.shape[:2], dtype=np.int64)
        for rgb, class_id in CLASS_COLORS.items():
            class_mask[np.all(rgb_mask == rgb, axis=-1)] = class_id
        return class_mask

    raise ValueError(f"Unsupported mask shape: {mask_np.shape}")


def build_reference_from_pixels(images_np, masks, fill_color, image_size):
    reference_stack = []
    for image_np, region_mask in zip(images_np, masks):
        reference = np.full((image_size, image_size, 3), fill_color, dtype=np.uint8)
        reference[region_mask] = image_np[region_mask]
        reference_stack.append(reference)

    return np.mean(np.stack(reference_stack, axis=0), axis=0).astype(np.uint8)


def compute_references(lusnar_dir, lusnar_masks_dir, image_size, n_samples):
    """Compute LuSNAR sky and terrain reference images for histogram matching."""
    from skimage import exposure

    del exposure

    pairs = collect_lusnar_pairs(Path(lusnar_dir), Path(lusnar_masks_dir))
    if not pairs:
        raise RuntimeError(
            f"No LuSNAR image/mask pairs found in {lusnar_dir}. Expected scene_XX "
            "folders with corresponding masks or Moon_*/image0/color + label folders."
        )

    if len(pairs) < n_samples:
        warnings.warn(
            f"Requested {n_samples} LuSNAR samples, but only found {len(pairs)} "
            "valid pairs. Using all available pairs.",
            RuntimeWarning,
        )
        sampled_pairs = pairs
    else:
        rng = random.Random(42)
        sampled_pairs = rng.sample(pairs, n_samples)

    images_np = []
    sky_masks = []
    terrain_masks = []
    sky_pixels = []
    terrain_pixels = []

    for image_path, mask_path in tqdm(sampled_pairs, desc="Building LuSNAR references"):
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.uint8)

        mask = Image.open(mask_path)
        mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
        mask_ids = mask_to_class_ids(mask)

        sky_mask = mask_ids == SKY_CLASS_ID
        terrain_mask = mask_ids != SKY_CLASS_ID

        images_np.append(image_np)
        sky_masks.append(sky_mask)
        terrain_masks.append(terrain_mask)
        if sky_mask.any():
            sky_pixels.append(image_np[sky_mask])
        if terrain_mask.any():
            terrain_pixels.append(image_np[terrain_mask])

    if not sky_pixels:
        raise RuntimeError("No LuSNAR sky pixels found in sampled masks (class id 4).")
    if not terrain_pixels:
        raise RuntimeError("No LuSNAR terrain pixels found in sampled masks.")

    mean_sky_color = np.mean(np.concatenate(sky_pixels, axis=0), axis=0).astype(np.uint8)
    mean_terrain_color = np.mean(np.concatenate(terrain_pixels, axis=0), axis=0).astype(
        np.uint8
    )

    sky_reference = build_reference_from_pixels(
        images_np, sky_masks, mean_sky_color, image_size
    )
    terrain_reference = build_reference_from_pixels(
        images_np, terrain_masks, mean_terrain_color, image_size
    )

    print(f"[INFO] LuSNAR pairs used: {len(sampled_pairs)}")
    print(f"[INFO] Mean sky color RGB: {mean_sky_color.tolist()}")
    print(f"[INFO] Mean terrain color RGB: {mean_terrain_color.tolist()}")

    return sky_reference, terrain_reference


def build_sky_mask(image_np, sky_threshold, upper_fraction):
    grayscale = image_np.mean(axis=2)
    height = image_np.shape[0]
    upper_boundary = int(height * upper_fraction)
    rows = np.arange(height)[:, None]
    return (rows < upper_boundary) & (grayscale < sky_threshold)


def apply_region_matching(image_pil, sky_ref, terrain_ref, sky_threshold, upper_fraction):
    from skimage import exposure

    img = np.asarray(image_pil, dtype=np.uint8)
    sky_mask = build_sky_mask(img, sky_threshold, upper_fraction)
    terrain_mask = ~sky_mask
    result = img.copy()

    if sky_mask.any():
        sky_matched = exposure.match_histograms(img, sky_ref, channel_axis=2)
        sky_matched = np.clip(sky_matched, 0, 255).astype(np.uint8)
        result[sky_mask] = sky_matched[sky_mask]

    terrain_matched = exposure.match_histograms(img, terrain_ref, channel_axis=2)
    terrain_matched = np.clip(terrain_matched, 0, 255).astype(np.uint8)
    result[terrain_mask] = terrain_matched[terrain_mask]

    return Image.fromarray(result)


def make_comparison_image(original, adjusted):
    comparison = Image.new("RGB", (original.width + adjusted.width, original.height))
    comparison.paste(original, (0, 0))
    comparison.paste(adjusted, (original.width, 0))
    return comparison


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Computing LuSNAR region references...")
    sky_ref, terrain_ref = compute_references(
        args.lusnar_dir,
        args.lusnar_dir,
        args.image_size,
        args.lusnar_samples,
    )

    image_paths = find_input_images(args.input_dir)
    print(f"[INFO] Found {len(image_paths)} input images in {args.input_dir}")

    processed = 0
    with_sky = 0
    terrain_only = 0

    for image_path in tqdm(image_paths, desc="Region matching real images"):
        try:
            original = Image.open(image_path)
            original = ImageOps.exif_transpose(original).convert("RGB")
            original = original.resize(
                (args.image_size, args.image_size), resample=Image.BILINEAR
            )
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping unreadable image {image_path}: {error}", RuntimeWarning)
            continue

        adjusted = apply_region_matching(
            original,
            sky_ref,
            terrain_ref,
            args.sky_threshold,
            args.upper_fraction,
        )
        adjusted.save(args.output_dir / image_path.name)
        make_comparison_image(original, adjusted).save(
            args.output_dir / f"{image_path.stem}_compare.png"
        )

        sky_mask = build_sky_mask(
            np.asarray(original, dtype=np.uint8), args.sky_threshold, args.upper_fraction
        )
        sky_count = int(sky_mask.sum())
        total_pixels = int(sky_mask.size)
        terrain_count = total_pixels - sky_count
        sky_percent = 100.0 * sky_count / total_pixels
        terrain_percent = 100.0 * terrain_count / total_pixels

        if sky_count > 0:
            with_sky += 1
        else:
            terrain_only += 1
        processed += 1

        print(
            f"{image_path.name} | sky pixels {sky_percent:.2f}% | "
            f"terrain pixels {terrain_percent:.2f}%"
        )

    print(
        f"[INFO] Summary: processed={processed}, "
        f"sky_detected={with_sky}, terrain_only={terrain_only}"
    )


if __name__ == "__main__":
    main()
