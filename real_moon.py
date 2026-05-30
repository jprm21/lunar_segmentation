#!/usr/bin/env python3
"""Run qualitative inference on real Moon images with a trained segmentation model.

The script saves one mosaic per input image: the resized RGB input on the left and
an alpha-blended prediction overlay on the right.
"""

import argparse
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
    return parser.parse_args()


def validate_args(args):
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


def prediction_to_overlay(image, prediction, palette, alpha):
    image_np = np.asarray(image, dtype=np.float32)
    color_mask = palette[prediction]
    overlay = ((1.0 - alpha) * image_np + alpha * color_mask.astype(np.float32)).clip(0, 255)
    return Image.fromarray(overlay.astype(np.uint8))


def make_mosaic(image, overlay):
    mosaic = Image.new("RGB", (image.width + overlay.width, image.height))
    mosaic.paste(image, (0, 0))
    mosaic.paste(overlay, (image.width, 0))
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

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.input}")
    print(f"[INFO] Images: {len(image_paths)} from {args.images}")
    print(f"[INFO] Inference size: {args.image_size}x{args.image_size}")
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Output: {args.output}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Real Moon inference"):
            image, tensor = preprocess_image(image_path, args.image_size)
            tensor = tensor.to(device)

            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            overlay = prediction_to_overlay(image, prediction, palette, args.alpha)
            mosaic = make_mosaic(image, overlay)
            mosaic.save(output_path_for(image_path, args.output, args.suffix))

    print("[INFO] Qualitative mosaics saved successfully.")


if __name__ == "__main__":
    main()
