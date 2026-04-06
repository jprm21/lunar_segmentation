from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, RandomResizedCrop

from src.utils.label_utils import rgb_to_class


ROCK_CLASS_ID = 1


def _resize_mask_tensor(mask_tensor, size_hw):
    mask_pil = Image.fromarray(mask_tensor.numpy().astype(np.uint8), mode="L")
    resized = TF.resize(mask_pil, size_hw, interpolation=InterpolationMode.NEAREST)
    return torch.as_tensor(np.array(resized, dtype=np.int64), dtype=torch.long)


def random_crop_with_class(
    image,
    mask,
    crop_size=256,
    target_classes=(1, 2),
    max_tries=10,
    focus_class=None,
    focus_prob=0.0,
    center_offset_range=None,
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
        if focus_class is not None and random.random() < focus_prob:
            focus_pixels = torch.nonzero(mask == focus_class, as_tuple=False)
            if focus_pixels.numel() > 0:
                cy, cx = focus_pixels[random.randrange(focus_pixels.shape[0])].tolist()
                if center_offset_range is not None:
                    dx = random.randint(center_offset_range[0], center_offset_range[1])
                    dy = random.randint(center_offset_range[0], center_offset_range[1])
                    cx += dx
                    cy += dy
                top = max(0, min(max_top, int(cy - crop_h // 2)))
                left = max(0, min(max_left, int(cx - crop_w // 2)))
                return top, left

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


def random_zoom_in(image, mask, output_size, scale=(0.5, 1.0), ratio=(0.9, 1.1)):
    """RandomResizedCrop-like zoom-in for image and segmentation mask."""
    i, j, h, w = RandomResizedCrop.get_params(
        image,
        scale=scale,
        ratio=ratio,
    )
    cropped_image = TF.resized_crop(
        image,
        i,
        j,
        h,
        w,
        size=(output_size, output_size),
        interpolation=InterpolationMode.BILINEAR,
    )
    mask_pil = Image.fromarray(mask.numpy().astype(np.uint8), mode="L")
    cropped_mask = TF.resized_crop(
        mask_pil,
        i,
        j,
        h,
        w,
        size=(output_size, output_size),
        interpolation=InterpolationMode.NEAREST,
    )
    mask_tensor = torch.as_tensor(np.array(cropped_mask, dtype=np.int64), dtype=torch.long)
    return cropped_image, mask_tensor


def _connected_components(binary_mask):
    """Return list of connected components as arrays of (y, x) coordinates."""
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    ys, xs = np.where(binary_mask)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue

        stack = [(y, x)]
        visited[y, x] = True
        comp_pixels = []

        while stack:
            cy, cx = stack.pop()
            comp_pixels.append((cy, cx))

            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        components.append(np.array(comp_pixels, dtype=np.int32))

    return components


def _extract_random_rock_patch(dataset, target_size):
    for _ in range(10):
        src_img_path, src_mask_path = random.choice(dataset.samples)

        src_image = Image.open(src_img_path).convert("RGB")
        src_mask_rgb = Image.open(src_mask_path).convert("RGB")

        src_image = TF.resize(
            src_image,
            (target_size, target_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        src_mask_rgb = TF.resize(
            src_mask_rgb,
            (target_size, target_size),
            interpolation=InterpolationMode.NEAREST,
        )

        src_mask = rgb_to_class(src_mask_rgb)
        rock_mask = src_mask == ROCK_CLASS_ID

        if not rock_mask.any():
            continue

        components = _connected_components(rock_mask)
        components = [comp for comp in components if comp.shape[0] >= 8]

        if not components:
            continue

        comp = random.choice(components)
        ys = comp[:, 0]
        xs = comp[:, 1]
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        patch_image = np.array(src_image, dtype=np.uint8)[y1:y2, x1:x2]
        patch_mask = rock_mask[y1:y2, x1:x2]
        return patch_image, patch_mask

    return None, None


def copy_paste_rocks(dataset, image, mask, p=0.5, min_rocks=1, max_rocks=5, scale_range=(0.8, 1.5)):
    if random.random() >= p:
        return image, mask, False

    image_np = np.array(image, dtype=np.uint8).copy()
    mask_np = mask.numpy().copy()
    h, w = mask_np.shape

    pasted_any = False
    num_rocks = random.randint(min_rocks, max_rocks)

    for _ in range(num_rocks):
        patch_image, patch_mask = _extract_random_rock_patch(dataset, target_size=dataset.image_size)
        if patch_image is None:
            continue

        ph, pw = patch_mask.shape
        scale = random.uniform(scale_range[0], scale_range[1])
        new_h = max(2, int(ph * scale))
        new_w = max(2, int(pw * scale))

        patch_img_pil = Image.fromarray(patch_image, mode="RGB")
        patch_mask_pil = Image.fromarray((patch_mask.astype(np.uint8) * 255), mode="L")

        patch_img_resized = np.array(
            TF.resize(patch_img_pil, (new_h, new_w), interpolation=InterpolationMode.BILINEAR),
            dtype=np.uint8,
        )
        patch_mask_resized = np.array(
            TF.resize(patch_mask_pil, (new_h, new_w), interpolation=InterpolationMode.NEAREST),
            dtype=np.uint8,
        ) > 0

        if new_h >= h or new_w >= w:
            continue

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        target_slice = np.s_[top:top + new_h, left:left + new_w]
        paste_mask = patch_mask_resized

        image_region = image_np[target_slice]
        image_region[paste_mask] = patch_img_resized[paste_mask]

        mask_region = mask_np[target_slice]
        mask_region[paste_mask] = ROCK_CLASS_ID

        pasted_any = True

    if not pasted_any:
        return image, mask, False

    image_out = Image.fromarray(image_np, mode="RGB")
    mask_out = torch.as_tensor(mask_np, dtype=torch.long)
    return image_out, mask_out, True


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
        self.augmentation_profile = augmentation_profile
        self.copy_paste_applied_count = 0

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check dataset structure.")

        print(f"[INFO] Found {len(self.samples)} samples")

    def reset_epoch_stats(self):
        self.copy_paste_applied_count = 0

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

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        image = TF.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        mask_rgb = TF.resize(
            mask_rgb,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )

        mask = torch.as_tensor(rgb_to_class(mask_rgb), dtype=torch.long)

        if self.augmentation_profile == "rock_boost":
            image, mask = random_crop_with_class(
                image=image,
                mask=mask,
                crop_size=self.crop_size,
                target_classes=(ROCK_CLASS_ID, 2),
                max_tries=max(15, self.max_crop_tries),
                focus_class=ROCK_CLASS_ID,
                focus_prob=0.8,
                center_offset_range=(-20, 20),
            )
            image, mask = random_zoom_in(
                image=image,
                mask=mask,
                output_size=self.image_size,
                scale=(0.5, 1.0),
                ratio=(0.9, 1.1),
            )
            image, mask, applied = copy_paste_rocks(
                dataset=self,
                image=image,
                mask=mask,
                p=0.5,
                min_rocks=1,
                max_rocks=5,
                scale_range=(0.8, 1.5),
            )
            if applied:
                self.copy_paste_applied_count += 1
        elif self.use_class_aware_crop:
            image, mask = random_crop_with_class(
                image=image,
                mask=mask,
                crop_size=self.crop_size,
                target_classes=self.target_classes,
                max_tries=self.max_crop_tries,
            )
            image = TF.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = _resize_mask_tensor(mask, (self.image_size, self.image_size))

        image = TF.to_tensor(image)

        assert image.shape[1:] == mask.shape, f"Image {image.shape}, Mask {mask.shape}"

        return image, mask
