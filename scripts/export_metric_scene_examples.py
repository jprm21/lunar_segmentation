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
    0: (187, 70, 156),
    1: (120, 0, 200),
    2: (232, 250, 80),
    3: (173, 69, 31),
    4: (34, 201, 248),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference for ONE model with controlled input size and image index."
    )

    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model_examples")

    parser.add_argument("--input-size", type=int, default=512)

    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Index of the image inside each scene (default: 0)",
    )

    parser.add_argument(
        "--scenes",
        type=int,
        nargs="+",
        default=[3, 5, 7],
    )

    return parser.parse_args()


def find_scene_sample_by_index(data_root, scene_id, image_index):
    rgb_dir = data_root / f"Moon_{scene_id}" / "image0" / "color"
    label_dir = data_root / f"Moon_{scene_id}" / "image0" / "label"

    if not rgb_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Scene {scene_id} not found")

    rgb_images = sorted(rgb_dir.glob("*.png"))

    if not rgb_images:
        raise RuntimeError(f"No images in scene {scene_id}")

    if image_index >= len(rgb_images):
        raise IndexError(
            f"Index {image_index} out of range (max {len(rgb_images)-1}) in scene {scene_id}"
        )

    rgb_path = rgb_images[image_index]
    label_path = label_dir / rgb_path.name

    if not label_path.exists():
        raise RuntimeError(f"Missing GT for {rgb_path.name}")

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

    overlay = (0.4 * image_np + 0.6 * pred_color).astype(np.uint8)

    return Image.fromarray(
        np.concatenate([image_np, gt_color, overlay], axis=1)
    )


def preprocess(image_pil, mask_pil, input_size, device):
    image_resized = TF.resize(
        image_pil,
        (input_size, input_size),
        interpolation=Image.BILINEAR,
    )

    mask_resized = TF.resize(
        mask_pil,
        (input_size, input_size),
        interpolation=Image.NEAREST,
    )

    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)

    image_np = (
        image_tensor.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)

    gt_mask_np = rgb_to_class(mask_resized)

    return image_tensor, image_np, gt_mask_np


def infer(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        pred_mask = torch.argmax(logits, dim=1)

    return pred_mask.squeeze(0).cpu().numpy()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print(f"[INFO] Loading model: {args.model_path}")

    model = UNetMobileNet(num_classes=5, pretrained=True)
    state_dict = torch.load(args.model_path, map_location=device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("[ERROR] Incompatible model architecture")
        print(e)
        return

    model.to(device)
    model.eval()

    run_name = args.model_path.stem
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene_id in args.scenes:
        rgb_path, label_path = find_scene_sample_by_index(
            args.data_root,
            scene_id,
            args.image_index,
        )

        image_pil = Image.open(rgb_path).convert("RGB")
        mask_pil = Image.open(label_path).convert("RGB")

        image_tensor, image_np, gt_mask_np = preprocess(
            image_pil,
            mask_pil,
            args.input_size,
            device,
        )

        pred_mask_np = infer(model, image_tensor)

        panel = build_horizontal_panel(
            image_np,
            gt_mask_np,
            pred_mask_np,
        )

        output_path = output_dir / f"scene_{scene_id:02d}_idx_{args.image_index}.png"
        panel.save(output_path)

        print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    main()
