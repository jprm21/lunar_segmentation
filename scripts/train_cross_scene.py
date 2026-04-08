import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision.transforms.functional as TF

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet
from src.utils.label_utils import CLASS_COLORS, rgb_to_class
from src.utils.losses import CombinedSegmentationLoss, load_class_weights


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
BATCH_SIZE = 4
EPOCHS = 40
LR = 3e-4
IMAGE_SIZE = 256
WEIGHT_DECAY = 1e-4

TRAIN_SCENES = [1, 2, 4, 6, 8, 9]
TEST_SCENES = [3, 5, 7]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-scene training for LuSNAR.")
    parser.add_argument(
        "--augmentation-profile",
        default="default",
        choices=["default", "rock_light"],
        help="Augmentation profile for the training dataset.",
    )
    parser.add_argument(
        "--save-aug-preview-dir",
        type=Path,
        default=None,
        help="Optional directory to save 5 augmentation preview samples.",
    )
    parser.add_argument(
        "--aug-preview-count",
        type=int,
        default=5,
        help="Number of augmentation previews to save when --save-aug-preview-dir is set.",
    )
    return parser.parse_args()


def build_train_sampler(dataset, image_size):
    """Build per-image weights to oversample samples with rare classes (1, 2)."""
    sample_weights = []

    for _, mask_path in dataset.samples:
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_rgb = TF.resize(
            mask_rgb,
            (image_size, image_size),
            interpolation=Image.NEAREST,
        )
        mask = torch.as_tensor(rgb_to_class(mask_rgb), dtype=torch.long)

        rare_pixels = (mask == 1).sum().item() + (mask == 2).sum().item()
        total_pixels = mask.numel()
        weight = 1.0 + 3.0 * (rare_pixels / total_pixels)
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )




def mask_to_color_preview(mask_tensor):
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)

    for rgb, class_id in CLASS_COLORS.items():
        color_mask[mask_np == class_id] = rgb

    return Image.fromarray(color_mask)

def save_augmentation_preview(dataset, output_dir, count):
    output_dir.mkdir(parents=True, exist_ok=True)

    total = min(count, len(dataset))
    for idx in range(total):
        original_image, original_mask, aug_image, aug_mask = dataset.get_preview_pair(idx)

        TF.to_pil_image(original_image).save(output_dir / f"sample_{idx:02d}_original_image.png")
        TF.to_pil_image(original_mask.to(torch.uint8)).save(output_dir / f"sample_{idx:02d}_original_mask_raw.png")
        mask_to_color_preview(original_mask).save(output_dir / f"sample_{idx:02d}_original_mask_color.png")
        TF.to_pil_image(aug_image).save(output_dir / f"sample_{idx:02d}_augmented_image.png")
        TF.to_pil_image(aug_mask.to(torch.uint8)).save(output_dir / f"sample_{idx:02d}_augmented_mask_raw.png")
        mask_to_color_preview(aug_mask).save(output_dir / f"sample_{idx:02d}_augmented_mask_color.png")

    print(f"[INFO] Saved {total} augmentation previews to: {output_dir}")


def compute_iou_per_class(pred, target, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return ious


def main():
    args = parse_args()

    print("Using device:", DEVICE)
    print("version 256, crop corregido imbalanced al 70%")
    print(f"[INFO] Active augmentation profile: {args.augmentation_profile}")

    train_dataset = LuSNARDataset(
        root_dir=DATA_ROOT,
        image_size=IMAGE_SIZE,
        scenes=TRAIN_SCENES,
        use_class_aware_crop=True,
        crop_size=max(32, (int(0.7 * IMAGE_SIZE) // 32) * 32),
        target_classes=(1, 2),
        max_crop_tries=10,
        augmentation_profile=args.augmentation_profile,
    )

    test_dataset = LuSNARDataset(
        root_dir=DATA_ROOT,
        image_size=IMAGE_SIZE,
        scenes=TEST_SCENES,
    )

    if args.save_aug_preview_dir is not None:
        save_augmentation_preview(
            dataset=train_dataset,
            output_dir=args.save_aug_preview_dir,
            count=args.aug_preview_count,
        )

    train_sampler = build_train_sampler(train_dataset, image_size=IMAGE_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetMobileNet(num_classes=NUM_CLASSES, pretrained=True)
    model.to(DEVICE)

    class_weights = load_class_weights("data/class_weights.json", device=DEVICE)
    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        ce_weight=1.0,
        focal_weight=0.5,
        dice_weight=0.5,
        gamma=2.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_miou = 0.0
    class_names = ["Regolith", "Crater", "Rock", "Mountain", "Sky"]

    for epoch in range(EPOCHS):
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
        test_loss = 0.0

        total_iou_per_class = [0.0] * NUM_CLASSES
        counts_per_class = [0] * NUM_CLASSES

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Testing"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)

                ious = compute_iou_per_class(preds, masks, NUM_CLASSES)

                for cls in range(NUM_CLASSES):
                    if not (ious[cls] != ious[cls]):
                        total_iou_per_class[cls] += ious[cls]
                        counts_per_class[cls] += 1

                test_loss += loss.item()

        test_loss /= len(test_loader)

        mean_iou_per_class = []
        for cls in range(NUM_CLASSES):
            if counts_per_class[cls] > 0:
                mean_iou_per_class.append(
                    total_iou_per_class[cls] / counts_per_class[cls]
                )
            else:
                mean_iou_per_class.append(float("nan"))

        mean_iou = sum(mean_iou_per_class) / NUM_CLASSES

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test  Loss: {test_loss:.4f}")
        print(f"Test  mIoU: {mean_iou:.4f}")
        print(f"LR: {current_lr:.6f}")

        for cls in range(NUM_CLASSES):
            print(f"IoU {class_names[cls]}: {mean_iou_per_class[cls]:.4f}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), "best_model_im_crop_256_70.pth")
            print(f"✅ Best model saved at epoch {epoch + 1} with mIoU: {mean_iou:.4f}")


if __name__ == "__main__":
    main()
