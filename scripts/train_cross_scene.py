import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision.transforms.functional as TF

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet
from src.utils.label_utils import rgb_to_class
from src.utils.losses import CombinedSegmentationLoss, load_class_weights

# -----------------------------
# Configuración
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

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


# -----------------------------
# Datasets
# -----------------------------
train_dataset = LuSNARDataset(
    root_dir=DATA_ROOT,
    image_size=IMAGE_SIZE,
    scenes=TRAIN_SCENES,
    use_class_aware_crop=True,
    crop_size=IMAGE_SIZE,
    target_classes=(1, 2),
    max_crop_tries=10,
)

test_dataset = LuSNARDataset(
    root_dir=DATA_ROOT,
    image_size=IMAGE_SIZE,
    scenes=TEST_SCENES,
)

train_sampler = build_train_sampler(train_dataset, image_size=IMAGE_SIZE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Modelo
# -----------------------------
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


# -----------------------------
# IoU por clase
# -----------------------------
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


# -----------------------------
# Training Loop
# -----------------------------
best_miou = 0.0
class_names = ["Regolith", "Rock", "Crater", "Mountain", "Sky"]

for epoch in range(EPOCHS):
    # ---- TRAIN ----
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

    # ---- TEST ----
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
                if not (ious[cls] != ious[cls]):  # check NaN
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

    # ---- PRINT RESULTADOS ----
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test  Loss: {test_loss:.4f}")
    print(f"Test  mIoU: {mean_iou:.4f}")
    print(f"LR: {current_lr:.6f}")

    for cls in range(NUM_CLASSES):
        print(f"IoU {class_names[cls]}: {mean_iou_per_class[cls]:.4f}")

    # ---- GUARDAR MEJOR MODELO ----
    if mean_iou > best_miou:
        best_miou = mean_iou
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Best model saved at epoch {epoch + 1} with mIoU: {mean_iou:.4f}")
