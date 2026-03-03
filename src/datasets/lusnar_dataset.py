from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms.functional as TF

from src.utils.label_utils import rgb_to_class


class LuSNARDataset(Dataset):
    def __init__(self, root_dir, image_size=256, transform=None, scenes=None):
        """
        root_dir: data/
        image_size: int (e.g. 256 or 512)
        scenes: list of ints (e.g. [1,2,4]) or None for all scenes
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.scenes = scenes  # ← NUEVO

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

        # --- Image to tensor ---
        image = TF.to_tensor(image)  # (3, H, W), float32 [0,1]

        # --- Sanity check (very important during development) ---
        assert image.shape[1:] == mask.shape, \
            f"Image {image.shape}, Mask {mask.shape}"

        return image, torch.as_tensor(mask, dtype=torch.long)
