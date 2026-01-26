from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch

from src.utils.label_utils import rgb_to_class


class LuSNARDataset(Dataset):
    def __init__(self, root_dir, camera="image0", transform=None):
        """
        root_dir: data/
        camera: image0 (image1 no tiene labels)
        """
        self.root_dir = Path(root_dir)
        self.camera = camera
        self.transform = transform
        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check dataset structure.")

    def _collect_samples(self):
        samples = []

        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            rgb_dir = scene_dir / self.camera / "color"
            label_dir = scene_dir / self.camera / "label"

            if not rgb_dir.exists() or not label_dir.exists():
                continue

            for rgb_path in sorted(rgb_dir.glob("*.png")):
                label_path = label_dir / rgb_path.name
                if label_path.exists():
                    samples.append((rgb_path, label_path))

        print(f"[INFO] Found {len(samples)} samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask = rgb_to_class(mask_rgb)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, torch.tensor(mask, dtype=torch.long)
