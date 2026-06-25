import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.train_cross_scene import build_train_sampler, compute_iou_per_class
from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet
from src.utils.losses import CombinedSegmentationLoss, load_class_weights


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CLASS_NAMES = ["Regolith", "Crater", "Rock", "Mountain", "Sky"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained LuSNAR segmentation model on real lunar scenes."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the pretrained .pth model checkpoint.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help='Path to the data root containing Moon_10, Moon_11, etc. (default: "data").',
    )
    parser.add_argument(
        "--train_scene",
        type=int,
        default=10,
        help="Scene number to use for fine-tuning training (default: 10).",
    )
    parser.add_argument(
        "--val_scene",
        type=int,
        default=11,
        help="Scene number to use for validation (default: 11).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        required=True,
        help="Square input size; must match the checkpoint training resolution.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of fine-tuning epochs (default: 30).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for fine-tuning (default: 3e-5).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--freeze_encoder",
        dest="freeze_encoder",
        action="store_true",
        help="Freeze MobileNetV2 encoder parameters.",
    )
    freeze_group.add_argument(
        "--no_freeze_encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Keep encoder parameters trainable.",
    )
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument(
        "--class_weights_path",
        type=Path,
        default=Path("data/class_weights.json"),
        help='Path to class weights JSON (default: "data/class_weights.json").',
    )
    parser.add_argument(
        "--output_name",
        type=Path,
        required=True,
        help="Filename/path for the best fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of segmentation classes (default: 5).",
    )
    parser.add_argument(
        "--oversample_rare_classes",
        action="store_true",
        help="Use train_cross_scene.py's WeightedRandomSampler on the real train scene.",
    )
    return parser.parse_args()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    return checkpoint


def load_pretrained_model(checkpoint_path, num_classes):
    model = UNetMobileNet(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


def freeze_encoder_parameters(model, freeze_encoder=True):
    if freeze_encoder:
        for name, param in model.named_parameters():
            # UNetMobileNet exposes the MobileNetV2 backbone as `self.encoder`.
            if "encoder" in name:
                param.requires_grad = False

    total_params = sum(param.numel() for param in model.parameters())
    frozen_params = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    trainable_params = total_params - frozen_params
    trainable_pct = (100.0 * trainable_params / total_params) if total_params else 0.0

    return total_params, frozen_params, trainable_params, trainable_pct


def main():
    args = parse_args()

    if args.image_size <= 0:
        raise ValueError("--image_size must be a positive integer")

    print("Using device:", DEVICE)

    # Reuse LuSNARDataset unchanged: the real Chang'e/Yutu-2 data follows the
    # same Moon_x/imageY/color + label layout, only with scene numbers 10/11.
    train_dataset = LuSNARDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        scenes=[args.train_scene],
        augmentation_profile="default",
    )

    val_dataset = LuSNARDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        scenes=[args.val_scene],
    )

    if args.oversample_rare_classes:
        train_sampler = build_train_sampler(train_dataset, image_size=args.image_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_pretrained_model(args.checkpoint, num_classes=args.num_classes)
    total_params, frozen_params, trainable_params, trainable_pct = freeze_encoder_parameters(
        model,
        freeze_encoder=args.freeze_encoder,
    )

    print("\nFine-tuning real lunar dataset")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Train scene: Moon_{args.train_scene}")
    print(f"Val scene: Moon_{args.val_scene}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Image size: {args.image_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Oversampling active: {args.oversample_rare_classes}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")

    class_weights = load_class_weights(args.class_weights_path, device=DEVICE)
    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        ce_weight=1.0,
        focal_weight=0.5,
        dice_weight=0.5,
        gamma=2.0,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        total_intersections = torch.zeros(args.num_classes, dtype=torch.float64, device=DEVICE)
        total_unions = torch.zeros(args.num_classes, dtype=torch.float64, device=DEVICE)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)

                intersections, unions = compute_iou_per_class(preds, masks, args.num_classes)
                total_intersections += intersections
                total_unions += unions

                val_loss += loss.item()

        val_loss /= len(val_loader)

        mean_iou_per_class = []
        present_ious = []
        for cls in range(args.num_classes):
            union = total_unions[cls].item()
            if union > 0:
                iou = (total_intersections[cls] / total_unions[cls]).item()
                mean_iou_per_class.append(iou)
                present_ious.append(iou)
            else:
                mean_iou_per_class.append(float("nan"))

        mean_iou = sum(present_ious) / len(present_ious) if present_ious else float("nan")

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   mIoU: {mean_iou:.4f}")
        print(f"LR: {current_lr:.6f}")

        for cls in range(args.num_classes):
            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
            print(f"IoU {class_name}: {mean_iou_per_class[cls]:.4f}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), args.output_name)
            print(f"✅ Best model saved at epoch {epoch + 1} with mIoU: {mean_iou:.4f}")


if __name__ == "__main__":
    main()
