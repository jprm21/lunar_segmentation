import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.unet_mobilenet import UNetMobileNet
from src.utils.label_utils import rgb_to_class


CLASS_COLORS = {
    0: (187, 70, 156),   # Regolith
    1: (120, 0, 200),    # Crater
    2: (232, 250, 80),   # Rock
    3: (173, 69, 31),    # Mountain
    4: (34, 201, 248),   # Sky
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export horizontal comparison images (RGB | GT | Prediction overlay) "
            "for scenes used in metrics."
        )
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=ROOT,
        help='Root directory where one or more "best_model.pth" files are stored.',
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model_examples")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--scenes",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Scenes used in metrics.",
    )
    return parser.parse_args()


def find_first_scene_sample(data_root, scene_id):
    rgb_dir = data_root / f"Moon_{scene_id}" / "image0" / "color"
    label_dir = data_root / f"Moon_{scene_id}" / "image0" / "label"

    if not rgb_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Scene {scene_id} not found at: {rgb_dir}")

    rgb_images = sorted(rgb_dir.glob("*.png"))
    if not rgb_images:
        raise RuntimeError(f"No RGB images found for scene {scene_id} at: {rgb_dir}")

    rgb_path = rgb_images[0]
    label_path = label_dir / rgb_path.name

    if not label_path.exists():
        raise RuntimeError(
            f"Ground truth for scene {scene_id} image {rgb_path.name} not found at: {label_path}"
        )

    return rgb_path, label_path


def colorize_mask(mask_np):
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask_np == class_id] = color

    return color_mask


def build_horizontal_panel(image_np, gt_mask_np, pred_mask_np):
    gt_color = colorize_mask(gt_mask_np)
    pred_color = colorize_mask(pred_mask_np)

    overlay = (0.6 * image_np + 0.4 * pred_color).astype(np.uint8)

    panel = np.concatenate([image_np, gt_color, overlay], axis=1)
    return Image.fromarray(panel)


def load_and_prepare_sample(rgb_path, label_path, image_size, device):
    image_pil = Image.open(rgb_path).convert("RGB")
    mask_pil = Image.open(label_path).convert("RGB")

    image_resized = TF.resize(
        image_pil,
        (image_size, image_size),
        interpolation=Image.BILINEAR,
    )
    mask_resized = TF.resize(
        mask_pil,
        (image_size, image_size),
        interpolation=Image.NEAREST,
    )

    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    gt_mask_np = rgb_to_class(mask_resized)

    return image_tensor, image_np, gt_mask_np


def infer_mask(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        pred_mask = torch.argmax(logits, dim=1)
    return pred_mask.squeeze(0).cpu().numpy()


def model_output_dir(models_root, model_path):
    relative_parent = model_path.parent.relative_to(models_root)
    relative_name = str(relative_parent).replace("/", "_")
    if not relative_name or relative_name == ".":
        relative_name = "root"
    return relative_name


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = sorted(args.models_root.rglob("best_model.pth"))
    if not model_paths:
        raise RuntimeError(
            f'No "best_model.pth" files found under: {args.models_root}'
        )

    print(f"[INFO] Found {len(model_paths)} model(s)")

    scene_samples = {}
    for scene_id in args.scenes:
        scene_samples[scene_id] = find_first_scene_sample(args.data_root, scene_id)

    for model_path in model_paths:
        print(f"\n[INFO] Processing model: {model_path}")

        model = UNetMobileNet(num_classes=5, pretrained=True)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        run_name = model_output_dir(args.models_root, model_path)
        model_out_dir = args.output_dir / run_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for scene_id, (rgb_path, label_path) in scene_samples.items():
            image_tensor, image_np, gt_mask_np = load_and_prepare_sample(
                rgb_path,
                label_path,
                image_size=args.image_size,
                device=device,
            )
            pred_mask_np = infer_mask(model, image_tensor)
            panel_image = build_horizontal_panel(image_np, gt_mask_np, pred_mask_np)

            output_path = model_out_dir / f"scene_{scene_id:02d}_{rgb_path.stem}.png"
            panel_image.save(output_path)

            print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
