import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet  # lo creamos luego
from src.metrics.iou import mean_iou                 # lo creamos luego


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------
    # Dataset & Dataloader
    # --------------------
    dataset = LuSNARDataset("data")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # --------------------
    # Model
    # --------------------
    num_classes = 5
    model = UNetMobileNet(num_classes=num_classes)
    model.to(device)

    # --------------------
    # Loss (with class weights)
    # --------------------
    class_weights = torch.tensor(
        [0.29, 15.2, 9.1, 6.2, 0.81],  # ← EJEMPLO, reemplaza por los tuyos
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --------------------
    # Optimizer
    # --------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --------------------
    # Training loop
    # --------------------
    epochs = 20

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg loss: {avg_loss:.4f}")

        # (Opcional luego)
        # iou = mean_iou(outputs, masks)
        # print(f"Mean IoU: {iou:.3f}")

    torch.save(model.state_dict(), "checkpoint.pth")
    print("Training finished. Model saved.")


if __name__ == "__main__":
    train()
