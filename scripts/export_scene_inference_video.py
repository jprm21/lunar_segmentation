import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV (cv2) is required. Install with: pip install opencv-python"
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.unet_mobilenet import UNetMobileNet


# RGB -> class id
CLASS_COLORS = {
    (187, 70, 156): 0,  # Lunar regolith
    (120, 0, 200): 1,   # Impact crater
    (232, 250, 80): 2,  # Rock
    (173, 69, 31): 3,   # Mountain
    (34, 201, 248): 4,  # Sky
}

# class id -> RGB
CLASS_ID_TO_COLOR = {class_id: rgb for rgb, class_id in CLASS_COLORS.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run inference over a scene and export a side-by-side video "
            "(input | predicted mask | overlay)."
        )
    )
    parser.add_argument("--scene-id", type=int, required=True, help="Scene id (e.g. 3)")
    parser.add_argument("--model-path", type=Path, default=ROOT / "best_model.pth")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Output .mp4 path. Default: outputs/scene_<id>_inference.mp4",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--input-size",
        type=int,
        default=384,
        help="Model inference size (image is resized to square)",
    )
    return parser.parse_args()


def colorize_prediction(pred_mask_np):
    h, w = pred_mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_ID_TO_COLOR.items():
        color_mask[pred_mask_np == class_id] = color

    return color_mask


def preprocess_image(image_pil, input_size, device):
    image_resized = image_pil.resize((input_size, input_size), resample=Image.BILINEAR)

    image_np = np.array(image_resized, dtype=np.uint8)
    image_tensor = (
        torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)

    return image_tensor, image_np


def infer_mask(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        pred_mask = torch.argmax(logits, dim=1)

    return pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)


def build_panel(image_np, pred_mask_np):
    pred_color = colorize_prediction(pred_mask_np)
    overlay = (0.4 * image_np + 0.6 * pred_color).astype(np.uint8)

    return np.concatenate([image_np, pred_color, overlay], axis=1)


def resolve_scene_images(data_root, scene_id):
    scene_color_dir = data_root / f"Moon_{scene_id}" / "image0" / "color"
    if not scene_color_dir.exists():
        raise FileNotFoundError(f"Scene folder not found: {scene_color_dir}")

    image_paths = sorted(scene_color_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in: {scene_color_dir}")

    return image_paths


def main():
    args = parse_args()

    output_video = args.output_video
    if output_video is None:
        output_video = ROOT / "outputs" / f"scene_{args.scene_id:02d}_inference.mp4"

    output_video.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model from: {args.model_path}")

    model = UNetMobileNet(num_classes=5, pretrained=True)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_paths = resolve_scene_images(args.data_root, args.scene_id)

    start = max(0, args.start_index)
    end = min(start + args.max_frames, len(image_paths))
    selected_paths = image_paths[start:end]

    if not selected_paths:
        raise RuntimeError("No images selected. Check --start-index and --max-frames.")

    frame_h = args.input_size
    frame_w = args.input_size * 3

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (frame_w, frame_h),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not open video writer. Check codec support (mp4v).")

    print(
        f"[INFO] Processing {len(selected_paths)} frames "
        f"(scene={args.scene_id}, start={start}, end={end - 1})"
    )

    for i, image_path in enumerate(selected_paths, start=1):
        image_pil = Image.open(image_path).convert("RGB")

        image_tensor, image_np = preprocess_image(
            image_pil=image_pil,
            input_size=args.input_size,
            device=device,
        )

        pred_mask_np = infer_mask(model, image_tensor)
        panel = build_panel(image_np, pred_mask_np)

        # cv2 expects BGR
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        writer.write(panel_bgr)

        if i % 50 == 0 or i == len(selected_paths):
            print(f"[INFO] Frame {i}/{len(selected_paths)}")

    writer.release()
    print(f"[OK] Video exported to: {output_video}")


if __name__ == "__main__":
    main()
