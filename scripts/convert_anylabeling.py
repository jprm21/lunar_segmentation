#!/usr/bin/env python3
"""Convert X-AnyLabeling JSON annotations to LuSNAR-compatible segmentation masks.

Reads LabelMe-format JSON files produced by X-AnyLabeling and generates PNG masks
matching the LuSNAR format. Output mask size matches the original image automatically.
Supports both output modes:
  - id:    single-channel PNG where each pixel value is the class ID (0-4)
  - color: RGB PNG where each pixel is the LuSNAR class color

Usage:
    python convert_anylabeling.py \
        --input_dir  path/to/folder_with_images_and_jsons \
        --output_dir path/to/masks_output \
        --mode       color     # or 'id'

Class mapping (LuSNAR official colors):
    0  regolith   #BB469C  (187,  70, 156)
    1  crater     #7800C8  (120,   0, 200)
    2  rock       #E8FA50  (232, 250,  80)
    3  mountain   #AD451F  (173,  69,  31)
    4  sky        #22C9F8  ( 34, 201, 248)

Unannotated pixels default to regolith (class 0).
Draw order: regolith -> mountain -> crater -> rock -> sky

Gap filling:
    Small regolith gaps between sky and mountain are filled automatically
    using morphological closing on the combined sky+mountain region.
    Kernel size is controlled with --gap_kernel (default: 5).
    Set --gap_kernel 0 to disable.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing
from tqdm import tqdm

# ── Class definitions ─────────────────────────────────────────────────────────

LABEL_TO_ID = {
    "regolith": 0,
    "crater":   1,
    "rock":     2,
    "mountain": 3,
    "sky":      4,
}

ID_TO_COLOR = {
    0: (187,  70, 156),  # regolith  #BB469C
    1: (120,   0, 200),  # crater    #7800C8
    2: (232, 250,  80),  # rock      #E8FA50
    3: (173,  69,  31),  # mountain  #AD451F
    4: ( 34, 201, 248),  # sky       #22C9F8
}

# Draw order: background first, foreground last
DRAW_ORDER = [0, 3, 1, 2, 4]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Class IDs considered "horizon region" for gap filling
HORIZON_CLASS_IDS = {3, 4}  # mountain and sky


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert X-AnyLabeling JSON annotations to LuSNAR mask PNGs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help=(
            "Directory containing both the original images and the .json "
            "annotation files from X-AnyLabeling (same folder)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where mask PNG files will be saved.",
    )
    parser.add_argument(
        "--mode",
        choices=["id", "color"],
        default="color",
        help=(
            "'color' -> RGB PNG with LuSNAR class colors (default, use for verification).\n"
            "'id'    -> single-channel PNG, pixel value = class ID (use for training)."
        ),
    )
    parser.add_argument(
        "--default_class",
        default="regolith",
        help="Class for pixels not covered by any annotation. Default: regolith.",
    )
    parser.add_argument(
        "--gap_kernel",
        type=int,
        default=5,
        help=(
            "Kernel size (px) for morphological closing used to fill small regolith "
            "gaps between sky and mountain masks. Use odd numbers. "
            "Set to 0 to disable. Default: 5."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    if args.default_class not in LABEL_TO_ID:
        raise ValueError(
            f"--default_class '{args.default_class}' is not valid. "
            f"Options: {list(LABEL_TO_ID.keys())}"
        )
    if args.gap_kernel < 0:
        raise ValueError("--gap_kernel must be >= 0")


# ── Size resolution ───────────────────────────────────────────────────────────

def resolve_image_size(json_path, data):
    """Read size from the original image next to the JSON, or fall back to JSON dims."""
    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            with Image.open(candidate) as img:
                return img.size  # (width, height)

    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if w and h:
        return int(w), int(h)

    warnings.warn(
        f"Could not determine image size for {json_path.name}. "
        "No matching image file found and JSON has no imageWidth/imageHeight. "
        "Defaulting to 384x384.",
        RuntimeWarning,
    )
    return 384, 384


# ── Gap filling ───────────────────────────────────────────────────────────────

def fill_horizon_gaps(id_mask, gap_kernel):
    """
    Fill small regolith gaps between sky and mountain using morphological closing.

    Works by:
    1. Building a binary mask of all horizon pixels (sky OR mountain)
    2. Applying closing to expand that region and fill small holes
    3. Where closing added new pixels (was regolith, now covered by closing),
       assign them the nearest horizon class using a simple vertical scan —
       pixels above the horizon boundary get sky, below get mountain.
    """
    if gap_kernel == 0:
        return id_mask

    horizon_mask = np.isin(id_mask, list(HORIZON_CLASS_IDS))

    struct = np.ones((gap_kernel, gap_kernel), dtype=bool)
    closed = binary_closing(horizon_mask, structure=struct)

    # Pixels that closing added (were regolith gaps, now should be horizon)
    new_pixels = closed & ~horizon_mask & (id_mask == 0)  # only fill regolith gaps

    if not new_pixels.any():
        return id_mask

    result = id_mask.copy()

    # For each new pixel, assign sky if it's above the centroid row of the
    # horizon region, mountain otherwise. Simple and avoids nearest-neighbor search.
    horizon_rows = np.where(horizon_mask.any(axis=1))[0]
    if len(horizon_rows) == 0:
        return id_mask

    # Find the boundary row between mountain and sky in the original mask
    # Sky tends to be in the upper portion, mountain just below it
    sky_rows    = np.where((id_mask == 4).any(axis=1))[0]
    mountain_rows = np.where((id_mask == 3).any(axis=1))[0]

    if len(sky_rows) > 0 and len(mountain_rows) > 0:
        # Boundary is between the last sky row and first mountain row
        boundary_row = (sky_rows.max() + mountain_rows.min()) // 2
    elif len(sky_rows) > 0:
        boundary_row = sky_rows.max()
    else:
        boundary_row = mountain_rows.min() if len(mountain_rows) > 0 else 0

    new_pixel_rows, new_pixel_cols = np.where(new_pixels)
    for r, c in zip(new_pixel_rows, new_pixel_cols):
        result[r, c] = 4 if r <= boundary_row else 3  # sky above, mountain below

    filled_count = int(new_pixels.sum())
    return result, filled_count



# ── Above-sky regolith correction ────────────────────────────────────────────

def fix_regolith_above_sky(id_mask, top_fraction=0.15):
    """
    Reclassify regolith pixels near the top of the image that are annotation gaps.

    Two cases handled:
    1. Regolith rows sandwiched between sky rows (sky above AND sky below
       within a small window) — handles gaps inside the sky region.
    2. Regolith rows in the top `top_fraction` of the image that have sky
       within a small window below them — handles thin lines above the sky.

    Regolith outside these two cases is left untouched.

    Returns (corrected_mask, number_of_pixels_fixed).
    """
    sky_rows = set(int(r) for r in np.where((id_mask == 4).any(axis=1))[0])
    if not sky_rows:
        return id_mask, 0

    result = id_mask.copy()
    fixed = 0
    height = id_mask.shape[0]
    window = 6
    top_limit = int(height * top_fraction)

    for row in range(height):
        row_pixels = id_mask[row, :]
        if not np.all(row_pixels == 0):
            continue

        has_sky_above = any(
            (row - d) in sky_rows for d in range(1, window + 1) if row - d >= 0
        )
        has_sky_below = any(
            (row + d) in sky_rows for d in range(1, window + 1) if row + d < height
        )

        # Case 1: sandwiched between sky rows
        if has_sky_above and has_sky_below:
            fixed += int((result[row, :] == 0).sum())
            result[row, result[row, :] == 0] = 4
            continue

        # Case 2: near top of image with sky just below
        if row < top_limit and has_sky_below:
            fixed += int((result[row, :] == 0).sum())
            result[row, result[row, :] == 0] = 4

    return result, fixed


# ── Core conversion ───────────────────────────────────────────────────────────

def points_to_polygon(points):
    return [tuple(pt) for pt in points]


def convert_json_to_mask(json_path, mode, default_class_id, gap_kernel):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    width, height = resolve_image_size(json_path, data)
    shapes = data.get("shapes", [])

    # Build ID mask filled with default class (regolith)
    id_mask = np.full((height, width), fill_value=default_class_id, dtype=np.uint8)
    id_image = Image.fromarray(id_mask, mode="L")
    draw = ImageDraw.Draw(id_image)

    shapes_by_class = {cid: [] for cid in DRAW_ORDER}
    skipped = []

    for shape in shapes:
        label      = shape.get("label", "").strip().lower()
        shape_type = shape.get("shape_type", "polygon")
        points     = shape.get("points", [])

        if label not in LABEL_TO_ID:
            skipped.append(label)
            continue
        if len(points) < 3:
            warnings.warn(
                f"Shape '{label}' in {json_path.name} has fewer than 3 points, skipped.",
                RuntimeWarning,
            )
            continue

        class_id = LABEL_TO_ID[label]

        if shape_type in ("polygon", "rectangle"):
            shapes_by_class[class_id].append(points_to_polygon(points))
        else:
            warnings.warn(
                f"Unsupported shape_type '{shape_type}' for label '{label}' "
                f"in {json_path.name}. Only polygon and rectangle are supported.",
                RuntimeWarning,
            )

    # Draw in defined order — foreground overwrites background
    for class_id in DRAW_ORDER:
        for polygon in shapes_by_class.get(class_id, []):
            draw.polygon(polygon, fill=int(class_id))

    if skipped:
        warnings.warn(
            f"{json_path.name}: unrecognized labels skipped: {sorted(set(skipped))}. "
            f"Valid labels: {list(LABEL_TO_ID.keys())}",
            RuntimeWarning,
        )

    id_mask = np.array(id_image, dtype=np.uint8)

    # Fill small regolith gaps between sky and mountain
    filled_count = 0
    if gap_kernel > 0:
        result = fill_horizon_gaps(id_mask, gap_kernel)
        if isinstance(result, tuple):
            id_mask, filled_count = result
        else:
            id_mask = result

    # Fix regolith pixels above the sky boundary
    id_mask, above_sky_fixed = fix_regolith_above_sky(id_mask)
    filled_count += above_sky_fixed

    if mode == "id":
        return Image.fromarray(id_mask, mode="L"), filled_count

    # color mode
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in ID_TO_COLOR.items():
        color_mask[id_mask == class_id] = color
    return Image.fromarray(color_mask, mode="RGB"), filled_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    default_class_id = LABEL_TO_ID[args.default_class]
    json_files = sorted(args.input_dir.glob("*.json"))

    if not json_files:
        raise RuntimeError(f"No .json files found in {args.input_dir}")

    gap_status = f"kernel={args.gap_kernel}px" if args.gap_kernel > 0 else "disabled"
    print(f"[INFO] Found {len(json_files)} annotation files in {args.input_dir}")
    print(f"[INFO] Output mode    : {args.mode}")
    print(f"[INFO] Default class  : {args.default_class} (id={default_class_id})")
    print(f"[INFO] Gap filling    : {gap_status}")
    print(f"[INFO] Output dir     : {args.output_dir}")

    converted    = 0
    failed       = 0
    total_filled = 0

    for json_path in tqdm(json_files, desc="Converting annotations"):
        try:
            mask, filled = convert_json_to_mask(
                json_path,
                mode=args.mode,
                default_class_id=default_class_id,
                gap_kernel=args.gap_kernel,
            )
            output_path = args.output_dir / f"{json_path.stem}.png"
            mask.save(output_path)
            converted    += 1
            total_filled += filled

        except Exception as error:  # noqa: BLE001
            warnings.warn(
                f"Failed to convert {json_path.name}: {error}", RuntimeWarning
            )
            failed += 1

    print(f"[INFO] Done — converted: {converted}, failed: {failed}")
    if args.gap_kernel > 0:
        print(f"[INFO] Total gap pixels filled: {total_filled}")
    print(f"[INFO] Masks saved to  : {args.output_dir}")


if __name__ == "__main__":
    main()
