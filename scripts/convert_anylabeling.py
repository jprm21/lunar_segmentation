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
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ── Class definitions ─────────────────────────────────────────────────────────

LABEL_TO_ID = {
    "regolith": 0,
    "crater":   1,
    "rock":     2,
    "mountain": 3,
    "sky":      4,
}

# Official LuSNAR hex codes converted to RGB
ID_TO_COLOR = {
    0: (187,  70, 156),  # regolith  #BB469C
    1: (120,   0, 200),  # crater    #7800C8
    2: (232, 250,  80),  # rock      #E8FA50
    3: (173,  69,  31),  # mountain  #AD451F
    4: ( 34, 201, 248),  # sky       #22C9F8
}

# Draw order: background first, foreground last
# regolith(0) -> mountain(3) -> crater(1) -> rock(2) -> sky(4)
DRAW_ORDER = [0, 3, 1, 2, 4]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    return parser.parse_args()


def validate_args(args):
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    if args.default_class not in LABEL_TO_ID:
        raise ValueError(
            f"--default_class '{args.default_class}' is not valid. "
            f"Options: {list(LABEL_TO_ID.keys())}"
        )


# ── Size resolution ───────────────────────────────────────────────────────────

def resolve_image_size(json_path, data):
    """
    Determine (width, height) for the output mask.

    Priority:
      1. Find the original image next to the JSON and read its actual size.
      2. Fall back to imageWidth/imageHeight stored in the JSON.
      3. Fall back to 384x384 with a warning.
    """
    # Try to find the original image file beside the JSON
    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            with Image.open(candidate) as img:
                return img.size  # (width, height)

    # Fall back to JSON-stored dimensions
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


# ── Core conversion ───────────────────────────────────────────────────────────

def points_to_polygon(points):
    return [tuple(pt) for pt in points]


def convert_json_to_mask(json_path, mode, default_class_id):
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

    if mode == "id":
        return Image.fromarray(id_mask, mode="L")

    # color mode: map each class ID to its LuSNAR RGB color
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in ID_TO_COLOR.items():
        color_mask[id_mask == class_id] = color
    return Image.fromarray(color_mask, mode="RGB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    default_class_id = LABEL_TO_ID[args.default_class]
    json_files = sorted(args.input_dir.glob("*.json"))

    if not json_files:
        raise RuntimeError(f"No .json files found in {args.input_dir}")

    print(f"[INFO] Found {len(json_files)} annotation files in {args.input_dir}")
    print(f"[INFO] Output mode    : {args.mode}")
    print(f"[INFO] Default class  : {args.default_class} (id={default_class_id})")
    print(f"[INFO] Output dir     : {args.output_dir}")

    converted = 0
    failed    = 0

    for json_path in tqdm(json_files, desc="Converting annotations"):
        try:
            mask = convert_json_to_mask(
                json_path,
                mode=args.mode,
                default_class_id=default_class_id,
            )
            output_path = args.output_dir / f"{json_path.stem}.png"
            mask.save(output_path)
            converted += 1

        except Exception as error:  # noqa: BLE001
            warnings.warn(
                f"Failed to convert {json_path.name}: {error}", RuntimeWarning
            )
            failed += 1

    print(f"[INFO] Done — converted: {converted}, failed: {failed}")
    print(f"[INFO] Masks saved to  : {args.output_dir}")


if __name__ == "__main__":
    main()
