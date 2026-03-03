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
# Función mIoU
# -----------------------------
def compute_iou(pred, target, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return sum(ious) / len(ious)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):
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
    test_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)

            total_iou += compute_iou(preds, masks, NUM_CLASSES)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    mean_iou = total_iou / len(test_loader)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test  Loss: {test_loss:.4f}")
    print(f"Test  mIoU: {mean_iou:.4f}")