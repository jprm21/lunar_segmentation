from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src.utils.label_utils import rgb_to_class


def random_crop_with_class(
    image,
    mask,
    crop_size=256,
    target_classes=(1, 2),
    max_tries=10,
):
    """
    Crop image/mask pair, preferring crops that contain target classes.

    image: PIL image (RGB)
    mask: torch.Tensor (H, W), class indices
    """
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = crop_size

    width, height = image.size
    if crop_h > height or crop_w > width:
        raise ValueError(
            f"crop_size {(crop_h, crop_w)} must be <= image size {(height, width)}"
        )

    max_top = height - crop_h
    max_left = width - crop_w

    def sample_crop_coords():
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        return top, left

    for _ in range(max_tries):
        top, left = sample_crop_coords()
        crop_mask = mask[top:top + crop_h, left:left + crop_w]
        if any((crop_mask == class_id).any().item() for class_id in target_classes):
            crop_image = TF.crop(image, top=top, left=left, height=crop_h, width=crop_w)
            return crop_image, crop_mask

    # Fallback: random crop even if no target classes appear
    top, left = sample_crop_coords()
    crop_image = TF.crop(image, top=top, left=left, height=crop_h, width=crop_w)
    crop_mask = mask[top:top + crop_h, left:left + crop_w]
    return crop_image, crop_mask


class LuSNARDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=256,
        transform=None,
        scenes=None,
        use_class_aware_crop=False,
        crop_size=256,
        target_classes=(1, 2),
        crop_target_probs=None,
        max_crop_tries=10,
        augmentation_profile="default",
    ):
        """
        root_dir: data/
        image_size: int (e.g. 256 or 512)
        scenes: list of ints (e.g. [1,2,4]) or None for all scenes
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.scenes = scenes
        self.use_class_aware_crop = use_class_aware_crop
        self.crop_size = crop_size
        self.target_classes = target_classes
        self.max_crop_tries = max_crop_tries
        self.crop_target_probs = crop_target_probs
        if self.crop_target_probs:
            total_prob = sum(self.crop_target_probs.values())
            if total_prob <= 0:
                raise ValueError("crop_target_probs must sum to a positive value")
            self.crop_target_probs = {
                cls_id: prob / total_prob
                for cls_id, prob in self.crop_target_probs.items()
            }
        self.augmentation_profile = augmentation_profile

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check dataset structure.")

        print(f"[INFO] Found {len(self.samples)} samples")

    def _collect_samples(self):
        samples = []

        for moon_dir in sorted(self.root_dir.glob("Moon_*")):
            scene_number = int(moon_dir.name.split("_")[-1])

            if self.scenes is not None and scene_number not in self.scenes:
                continue

            cam_dir = moon_dir / "image0"
            rgb_dir = cam_dir / "color"
            label_dir = cam_dir / "label"

            if not rgb_dir.exists() or not label_dir.exists():
                continue

            for rgb_path in sorted(rgb_dir.glob("*.png")):
                label_path = label_dir / rgb_path.name
                if label_path.exists():
                    samples.append((rgb_path, label_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_preprocessed_pair(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        image = TF.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=Image.BILINEAR,
        )

        mask_rgb = TF.resize(
            mask_rgb,
            (self.image_size, self.image_size),
            interpolation=Image.NEAREST,
        )

        mask = rgb_to_class(mask_rgb)

        if self.use_class_aware_crop:
            mask = torch.as_tensor(mask, dtype=torch.long)

            target_classes = self.target_classes
            if self.crop_target_probs:
                classes = list(self.crop_target_probs.keys())
                probs = list(self.crop_target_probs.values())
                target_classes = (random.choices(classes, weights=probs, k=1)[0],)

            image, mask = random_crop_with_class(
                image=image,
                mask=mask,
                crop_size=self.crop_size,
                target_classes=target_classes,
                max_tries=self.max_crop_tries,
            )
        else:
            mask = torch.as_tensor(mask, dtype=torch.long)

        image = TF.to_tensor(image)
        return image, mask

    def _apply_augmentations(self, image, mask):
        if self.augmentation_profile != "rock_light":
            return image, mask

        if random.random() < 0.3:
            i, j, h, w = T.RandomResizedCrop.get_params(
                image,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
            )
            output_size = [image.shape[-2], image.shape[-1]]
            image = TF.resized_crop(
                image,
                top=i,
                left=j,
                height=h,
                width=w,
                size=output_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resized_crop(
                mask.unsqueeze(0).float(),
                top=i,
                left=j,
                height=h,
                width=w,
                size=output_size,
                interpolation=InterpolationMode.NEAREST,
            ).squeeze(0).long()

        if random.random() < 0.3:
            angle = random.uniform(-10.0, 10.0)
            image = TF.rotate(
                image,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = TF.rotate(
                mask.unsqueeze(0).float(),
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
                fill=255,
            ).squeeze(0).long()

        if random.random() < 0.2:
            brightness_factor = random.uniform(0.95, 1.05)
            contrast_factor = random.uniform(0.9, 1.1)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        return image, mask

    def get_preview_pair(self, idx):
        image, mask = self._load_preprocessed_pair(idx)
        original_image = image.clone()
        original_mask = mask.clone()
        aug_image, aug_mask = self._apply_augmentations(image, mask)
        return original_image, original_mask, aug_image, aug_mask

    def __getitem__(self, idx):
        image, mask = self._load_preprocessed_pair(idx)

        image, mask = self._apply_augmentations(image, mask)

        assert image.shape[1:] == mask.shape, \
            f"Image {image.shape}, Mask {mask.shape}"

        return image, mask
