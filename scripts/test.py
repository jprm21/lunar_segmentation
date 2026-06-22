import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.unet_mobilenet import UNetMobileNet
from src.utils.label_utils import CLASS_COLORS, NUM_CLASSES, rgb_to_class


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Regolith", "Crater", "Rock", "Mountain", "Sky"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Final evaluation for a saved LuSNAR segmentation model. "
            "Expects --test_images to contain image0/color and image0/labels "
            "(or image0/label) with matching file names."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to the .pth model checkpoint.")
    parser.add_argument(
        "--test_images",
        type=Path,
        required=True,
        help="Folder that contains image0/color and image0/labels with PNG masks.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Square resize size used before inference (default: 1024).",
    )
    parser.add_argument(
        "--save-segmentation-dir",
        type=Path,
        default=None,
        help=(
            "Optional output folder for 3-panel mosaics: resized input, "
            "ground truth, and prediction overlay."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"Number of output classes used by the model (default: {NUM_CLASSES}).",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Instantiate MobileNetV2 without ImageNet weights before loading the checkpoint.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help="Label value ignored when computing metrics (default: 255).",
    )
    parser.add_argument(
        "--sixteen-bit-scale",
        choices=["per-image", "fixed"],
        default="per-image",
        help=(
            "How to rescale 16-bit grayscale images (mode 'I;16') to 8-bit RGB. "
            "'per-image' scales by that image's own max value (default). "
            "'fixed' scales by the theoretical max of 65535."
        ),
    )
    return parser.parse_args()


def resolve_test_dirs(test_root):
    image0_dir = test_root / "image0"
    color_dir = image0_dir / "color"
    label_dir = image0_dir / "labels"

    if not label_dir.exists():
        label_dir = image0_dir / "label"

    if not color_dir.exists():
        raise FileNotFoundError(f"Missing color folder: {color_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing labels folder: {image0_dir / 'labels'} or {image0_dir / 'label'}")

    return color_dir, label_dir


def collect_samples(test_root):
    color_dir, label_dir = resolve_test_dirs(test_root)
    samples = []

    for image_path in sorted(color_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / image_path.name
        if label_path.exists():
            samples.append((image_path, label_path))
        else:
            print(f"[WARN] Missing label for {image_path.name}; skipping")

    if not samples:
        raise RuntimeError(f"No labeled samples found in {color_dir} and {label_dir}")

    return samples


def load_image_safe(path, sixteen_bit_scale="per-image"):
    """
    Load an image as RGB, correctly handling 16-bit grayscale sources.

    PIL's default .convert("RGB") on mode "I;16" images (0-65535 range) does
    not rescale to the 8-bit range correctly, producing washed-out/near-white
    images. This function detects that mode and rescales explicitly before
    converting, so pixel values returned are meaningful 0-255 RGB.

    Parameters
    ----------
    path : Path or str
        Image file path.
    sixteen_bit_scale : str
        'per-image' -> scale by this image's own max value (adapts to dynamic
                       range actually used by this specific image).
        'fixed'     -> scale by the theoretical max of 65535 (consistent
                       scaling across all 16-bit images, but may look dim if
                       the source doesn't use the full dynamic range).

    Returns
    -------
    PIL.Image in RGB mode.
    """
    img = Image.open(path)

    if img.mode == "I;16":
        arr = np.asarray(img).astype(np.float32)

        if sixteen_bit_scale == "fixed":
            denom = 65535.0
        else:  # "per-image"
            denom = float(arr.max()) if arr.max() > 0 else 1.0

        arr_8bit = np.clip(arr / denom * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_8bit, mode="L").convert("RGB")
    else:
        img = img.convert("RGB")

    return img


def compute_iou_per_class(pred, target, num_classes, ignore_index=255):
    intersections = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)

    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersections[cls] = (pred_inds & target_inds).sum()
        unions[cls] = (pred_inds | target_inds).sum()

    return intersections, unions


def colorize_mask(mask_np):
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for rgb, class_id in CLASS_COLORS.items():
        color_mask[mask_np == class_id] = rgb
    return color_mask


def build_mosaic(image_np, gt_mask_np, pred_mask_np, overlay_alpha):
    gt_color = colorize_mask(gt_mask_np)
    pred_color = colorize_mask(pred_mask_np)
    overlay = ((1.0 - overlay_alpha) * image_np + overlay_alpha * pred_color).astype(np.uint8)
    return Image.fromarray(np.concatenate([image_np, gt_color, overlay], axis=1))


def load_model(model_path, num_classes, pretrained_backbone, device):
    model = UNetMobileNet(num_classes=num_classes, pretrained=pretrained_backbone)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def preprocess(image_path, label_path, image_size, device, sixteen_bit_scale="per-image"):
    image = load_image_safe(image_path, sixteen_bit_scale=sixteen_bit_scale)
    label = Image.open(label_path).convert("RGB")

    image_resized = TF.resize(image, (image_size, image_size), interpolation=Image.BILINEAR)
    label_resized = TF.resize(label, (image_size, image_size), interpolation=Image.NEAREST)

    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
    image_np = (np.asarray(image_resized)).astype(np.uint8)
    target_np = rgb_to_class(label_resized)
    target_tensor = torch.as_tensor(target_np, dtype=torch.long, device=device)

    return image_tensor, image_np, target_np, target_tensor


def format_metric(value):
    return "nan" if np.isnan(value) else f"{value:.4f}"


def main():
    args = parse_args()

    if args.image_size <= 0:
        raise ValueError("--image-size must be a positive integer")

    samples = collect_samples(args.test_images)
    print(f"Using device: {DEVICE}")
    print(f"[INFO] Loading model: {args.model}")
    print(f"[INFO] Found {len(samples)} labeled test images")
    print(f"[INFO] Resize before inference: {args.image_size}x{args.image_size}")
    print(f"[INFO] 16-bit image scaling mode: {args.sixteen_bit_scale}")

    model = load_model(
        model_path=args.model,
        num_classes=args.num_classes,
        pretrained_backbone=not args.no_pretrained_backbone,
        device=DEVICE,
    )

    if args.save_segmentation_dir is not None:
        args.save_segmentation_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving mosaics to: {args.save_segmentation_dir}")

    total_intersections = torch.zeros(args.num_classes, dtype=torch.float64, device=DEVICE)
    total_unions = torch.zeros(args.num_classes, dtype=torch.float64, device=DEVICE)
    correct_pixels = torch.zeros((), dtype=torch.float64, device=DEVICE)
    valid_pixels = torch.zeros((), dtype=torch.float64, device=DEVICE)

    sixteen_bit_count = 0

    with torch.no_grad():
        for image_path, label_path in tqdm(samples, desc="Testing"):
            # Track how many 16-bit images were encountered, for the summary
            with Image.open(image_path) as raw_img:
                if raw_img.mode == "I;16":
                    sixteen_bit_count += 1

            image_tensor, image_np, target_np, target_tensor = preprocess(
                image_path=image_path,
                label_path=label_path,
                image_size=args.image_size,
                device=DEVICE,
                sixteen_bit_scale=args.sixteen_bit_scale,
            )

            logits = model(image_tensor)
            pred_tensor = torch.argmax(logits, dim=1).squeeze(0)

            intersections, unions = compute_iou_per_class(
                pred_tensor,
                target_tensor,
                args.num_classes,
                ignore_index=args.ignore_index,
            )
            total_intersections += intersections
            total_unions += unions

            valid_mask = target_tensor != args.ignore_index
            correct_pixels += (pred_tensor[valid_mask] == target_tensor[valid_mask]).sum()
            valid_pixels += valid_mask.sum()

            if args.save_segmentation_dir is not None:
                pred_np = pred_tensor.cpu().numpy().astype(np.uint8)
                mosaic = build_mosaic(image_np, target_np, pred_np, overlay_alpha=0.6)
                mosaic.save(args.save_segmentation_dir / f"{image_path.stem}_mosaic.png")

    ious = []
    present_ious = []
    for cls in range(args.num_classes):
        union = total_unions[cls].item()
        if union > 0:
            iou = (total_intersections[cls] / total_unions[cls]).item()
            present_ious.append(iou)
        else:
            iou = float("nan")
        ious.append(iou)

    miou = sum(present_ious) / len(present_ious) if present_ious else float("nan")
    global_accuracy = (correct_pixels / valid_pixels).item() if valid_pixels.item() > 0 else float("nan")

    print("\nFinal test metrics")
    print(f"Images: {len(samples)}")
    if sixteen_bit_count > 0:
        print(f"[INFO] 16-bit images detected and rescaled: {sixteen_bit_count}")
    print(f"mIoU: {format_metric(miou)}")
    print(f"Global pixel accuracy: {format_metric(global_accuracy)}")

    for cls, iou in enumerate(ious):
        class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        print(f"IoU {class_name}: {format_metric(iou)}")


if __name__ == "__main__":
    main()
