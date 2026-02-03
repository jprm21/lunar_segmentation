import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet
#from src.utils.losses import weighted_cross_entropy
from src.utils.label_utils import rgb_to_class
from src.utils.losses import load_class_weights




PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


# -----------------------------
# Configuración
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_CLASSES = 5
BATCH_SIZE = 4          # pequeño para sanity
EPOCHS = 1              # 1 o 2
LR = 1e-4
IMAGE_SIZE = 256

# -----------------------------
# Dataset
# -----------------------------
train_dataset = LuSNARDataset(
    root_dir=DATA_ROOT,
    image_size=IMAGE_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# -----------------------------
# Modelo
# -----------------------------
model = UNetMobileNet(num_classes=NUM_CLASSES)
model.to(DEVICE)

# -----------------------------
# Pesos de clases
# -----------------------------
class_weights = load_class_weights("data/class_weights.json")
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training loop
# -----------------------------
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

print("✅ Sanity training finished")
