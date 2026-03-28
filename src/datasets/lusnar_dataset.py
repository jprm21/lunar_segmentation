from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import random
import torch
import torchvision.transforms.functional as TF

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
        max_crop_tries=10,
    ):
        """
        root_dir: data/
        image_size: int (e.g. 256 or 512)
        scenes: list of ints (e.g. [1,2,4]) or None for all scenes
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.scenes = scenes  # ← NUEVO
        self.use_class_aware_crop = use_class_aware_crop
        self.crop_size = crop_size
        self.target_classes = target_classes
        self.max_crop_tries = max_crop_tries

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check dataset structure.")

        print(f"[INFO] Found {len(self.samples)} samples")

    def _collect_samples(self):
        samples = []

        for moon_dir in sorted(self.root_dir.glob("Moon_*")):
            # Extraer número de escena
            scene_number = int(moon_dir.name.split("_")[-1])

            # Filtrar si scenes está definido
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

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # --- Load ---
        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        # --- Resize (CRITICAL: same size, different interpolation) ---
        image = TF.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=Image.BILINEAR
        )

        mask_rgb = TF.resize(
            mask_rgb,
            (self.image_size, self.image_size),
            interpolation=Image.NEAREST
        )

        # --- Convert mask RGB → class indices ---
        mask = rgb_to_class(mask_rgb)  # (H, W), int

        if self.use_class_aware_crop:
            image, mask = random_crop_with_class(
                image=image,
                mask=torch.as_tensor(mask, dtype=torch.long),
                crop_size=self.crop_size,
                target_classes=self.target_classes,
                max_tries=self.max_crop_tries,
            )
        else:
            mask = torch.as_tensor(mask, dtype=torch.long)

        # --- Image to tensor ---
        image = TF.to_tensor(image)  # (3, H, W), float32 [0,1]

        # --- Sanity check (very important during development) ---
        assert image.shape[1:] == mask.shape, \
            f"Image {image.shape}, Mask {mask.shape}"

        return image, mask
