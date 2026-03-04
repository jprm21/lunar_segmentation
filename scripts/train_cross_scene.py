import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet
from src.utils.losses import load_class_weights

# -----------------------------
# Configuración
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_CLASSES = 5
BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-4
IMAGE_SIZE = 256

TRAIN_SCENES = [1,2,4,6,8,9]
TEST_SCENES  = [3,5,7]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# -----------------------------
# Datasets
# -----------------------------
train_dataset = LuSNARDataset(
    root_dir=DATA_ROOT,
    image_size=IMAGE_SIZE,
    scenes=TRAIN_SCENES
)

test_dataset = LuSNARDataset(
    root_dir=DATA_ROOT,
    image_size=IMAGE_SIZE,
    scenes=TEST_SCENES
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Modelo
# -----------------------------
model = UNetMobileNet(num_classes=NUM_CLASSES, pretrained=False)
model.to(DEVICE)

class_weights = load_class_weights("data/class_weights.json")
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# IoU por clase
# -----------------------------
def compute_iou_per_class(pred, target, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))
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
            mean_iou_per_class.append(float('nan'))

    mean_iou = sum(mean_iou_per_class) / NUM_CLASSES

    # ---- PRINT RESULTADOS ----
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test  Loss: {test_loss:.4f}")
    print(f"Test  mIoU: {mean_iou:.4f}")

    for cls in range(NUM_CLASSES):
        print(f"IoU {class_names[cls]}: {mean_iou_per_class[cls]:.4f}")

    # ---- GUARDAR MEJOR MODELO ----
    if mean_iou > best_miou:
        best_miou = mean_iou
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1} with mIoU: {mean_iou:.4f}")