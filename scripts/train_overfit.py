import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet
from src.utils.losses import load_class_weights

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# -----------------------------
# Configuración
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_CLASSES = 5
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-3
IMAGE_SIZE = 256
OVERFIT_SAMPLES = 16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "overfit_outputs"
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Dataset completo
# -----------------------------
full_dataset = LuSNARDataset(
    root_dir=DATA_ROOT,
    image_size=IMAGE_SIZE
)

print(f"[INFO] Dataset size: {len(full_dataset)}")

# -----------------------------
# Subset de 16 imágenes
# -----------------------------
subset_indices = list(range(OVERFIT_SAMPLES))
train_dataset = Subset(full_dataset, subset_indices)

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
# Loss con pesos
# -----------------------------
class_weights = load_class_weights("data/class_weights.json")
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Imagen fija para monitoreo
# -----------------------------
fixed_img, fixed_mask = train_dataset[0]
fixed_img = fixed_img.unsqueeze(0).to(DEVICE)

# -----------------------------
# Training loop
# -----------------------------
print("\n[START] Overfit test on 16 samples\n")

for epoch in range(EPOCHS):
    model.train()
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

    # -------------------------
    # Guardar predicción fija
    # -------------------------
    model.eval()
    with torch.no_grad():
        pred = model(fixed_img)
        pred = torch.argmax(pred, dim=1).float() / (NUM_CLASSES - 1)

        save_image(
            pred.unsqueeze(1),
            OUT_DIR / f"epoch_{epoch:03d}.png"
        )

    print(f"Epoch [{epoch+1:03d}/{EPOCHS}] - Loss: {avg_loss:.6f}")

print("\n✅ Overfit test finished")
print(f"Predictions saved in: {OUT_DIR}")
