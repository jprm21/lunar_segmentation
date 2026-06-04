#!/usr/bin/env python3
"""Run qualitative inference on real Moon images with a trained segmentation model.

The script saves one mosaic per input image: the resized RGB input on the left,
the model input in the center, and an alpha-blended prediction overlay on the right.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.unet_mobilenet import UNetMobileNet
from src.utils.label_utils import CLASS_COLORS, NUM_CLASSES

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run inference over every image in a folder and save qualitative "
            "input/overlay mosaics."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the trained .pth model weights or checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Folder where qualitative mosaics will be written.",
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Folder containing the images to evaluate.",
    )
    parser.add_argument(
        "--image_size",
        required=True,
        type=int,
        help="Square size in pixels used to resize each image before inference.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Prediction overlay opacity in [0, 1]. Default: 0.45.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, e.g. 'cuda' or 'cpu'. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--suffix",
        default="_real_moon",
        help="Suffix added to every saved mosaic filename before .png.",
    )
    parser.add_argument(
        "--lusnar_dir",
        type=Path,
        default=None,
        help=(
            "Path to the LuSNAR dataset root. When provided, histogram matching "
            "is applied to each input image before inference."
        ),
    )
    parser.add_argument(
        "--lusnar_samples",
        type=int,
        default=500,
        help=(
            "Number of LuSNAR images to randomly sample for computing the "
            "reference histogram. Default: 500."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.lusnar_dir is not None and not args.lusnar_dir.exists():
        raise FileNotFoundError(f"LuSNAR dataset root not found: {args.lusnar_dir}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.input}")
    if not args.images.is_dir():
        raise NotADirectoryError(f"Images folder not found: {args.images}")
    if args.image_size <= 0:
        raise ValueError("--image_size must be a positive integer")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")


def find_images(images_dir):
    image_paths = [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_paths:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RuntimeError(f"No supported images found in {images_dir}. Supported: {supported}")
    return image_paths


def normalize_state_dict_keys(state_dict):
    """Remove common training wrappers such as DataParallel's 'module.' prefix."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    return normalized


def extract_state_dict(checkpoint):
    """Accept either a raw state_dict or a checkpoint dict containing one."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format: expected a dict or state_dict")

    candidate_keys = (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "weights",
    )
    for key in candidate_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return normalize_state_dict_keys(checkpoint[key])

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return normalize_state_dict_keys(checkpoint)

    raise KeyError(
        "Could not find model weights in checkpoint. Expected a raw state_dict or "
        "one of: state_dict, model_state_dict, model, net, weights."
    )


def infer_num_classes(state_dict):
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        return NUM_CLASSES
    return int(classifier_weight.shape[0])


def load_model(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    num_classes = infer_num_classes(state_dict)

    model = UNetMobileNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes


def build_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for rgb, class_id in CLASS_COLORS.items():
        if class_id < num_classes:
            palette[class_id] = rgb

    if num_classes > len(CLASS_COLORS):
        rng = np.random.default_rng(seed=42)
        for class_id in range(len(CLASS_COLORS), num_classes):
            palette[class_id] = rng.integers(0, 256, size=3, dtype=np.uint8)

    return palette


def preprocess_image(image_path, image_size):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    tensor = TF.to_tensor(image).unsqueeze(0)
    return image, tensor


def compute_lusnar_reference(lusnar_dir, image_size, n_samples) -> np.ndarray:
    from skimage import exposure

    del exposure

    supported = {".jpg", ".jpeg", ".png"}
    image_paths = []
    for moon_dir in sorted(lusnar_dir.glob("Moon_*")):
        if not moon_dir.is_dir():
            continue
        for image_dir in sorted(moon_dir.iterdir()):
            rgb_dir = image_dir / "color"
            if not rgb_dir.is_dir():
                continue
            image_paths.extend(
                path
                for path in sorted(rgb_dir.iterdir())
                if path.is_file() and path.suffix.lower() in supported
            )

    if not image_paths:
        raise RuntimeError(f"No LuSNAR RGB images found in {lusnar_dir}")

    if len(image_paths) < n_samples:
        print(
            f"[WARN] Requested {n_samples} LuSNAR samples, but only found "
            f"{len(image_paths)}. Using all available images."
        )
        sampled_paths = image_paths
    else:
        rng = random.Random(42)
        sampled_paths = rng.sample(image_paths, n_samples)

    images_np = []
    for image_path in sampled_paths:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
        images_np.append(np.asarray(image, dtype=np.uint8))

    reference = np.mean(np.stack(images_np, axis=0), axis=0).astype(np.uint8)
    print(f"[INFO] LuSNAR reference computed from {len(sampled_paths)} images")
    return reference


def apply_histogram_matching(image_pil, reference_np) -> Image.Image:
    from skimage import exposure

    image_np = np.asarray(image_pil, dtype=np.uint8)
    matched = exposure.match_histograms(image_np, reference_np, channel_axis=2)
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))


def prediction_to_overlay(image, prediction, palette, alpha):
    image_np = np.asarray(image, dtype=np.float32)
    color_mask = palette[prediction]
    overlay = ((1.0 - alpha) * image_np + alpha * color_mask.astype(np.float32)).clip(0, 255)
    return Image.fromarray(overlay.astype(np.uint8))


def make_mosaic(original, matched, overlay):
    mosaic = Image.new(
        "RGB",
        (original.width + matched.width + overlay.width, original.height),
    )
    mosaic.paste(original, (0, 0))
    mosaic.paste(matched, (original.width, 0))
    mosaic.paste(overlay, (original.width + matched.width, 0))
    return mosaic


def output_path_for(input_path, output_dir, suffix):
    safe_suffix = suffix if suffix else ""
    return output_dir / f"{input_path.stem}{safe_suffix}.png"


def main():
    args = parse_args()
    validate_args(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(args.images)
    model, num_classes = load_model(args.input, device)
    palette = build_palette(num_classes)
    reference = None
    if args.lusnar_dir is not None:
        reference = compute_lusnar_reference(
            args.lusnar_dir, args.image_size, args.lusnar_samples
        )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.input}")
    print(f"[INFO] Images: {len(image_paths)} from {args.images}")
    print(f"[INFO] Inference size: {args.image_size}x{args.image_size}")
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Output: {args.output}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Real Moon inference"):
            image, _ = preprocess_image(image_path, args.image_size)
            if reference is not None:
                matched = apply_histogram_matching(image, reference)
                print("  [hist-match] applied")
                matched.save(args.output / f"{image_path.stem}_matched.png")
            else:
                matched = image

            tensor = TF.to_tensor(matched).unsqueeze(0).to(device)

            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            overlay = prediction_to_overlay(matched, prediction, palette, args.alpha)
            mosaic = make_mosaic(image, matched, overlay)
            mosaic.save(output_path_for(image_path, args.output, args.suffix))

    print("[INFO] Qualitative mosaics saved successfully.")


if __name__ == "__main__":
    main()
