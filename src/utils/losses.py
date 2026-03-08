import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_class_weights(path, device="cpu"):
    """
    Load class weights from a JSON file and return a torch tensor.
    """
    with open(path, "r") as f:
        weights = json.load(f)

    # Ensure order by class index
    weights = [weights[str(i)] for i in range(len(weights))]

    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        cardinality = torch.sum(probs + target_oh, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    CE + Focal + Dice loss.
    Useful for highly imbalanced semantic segmentation datasets.
    """

    def __init__(self, class_weights=None, ce_weight=1.0, focal_weight=0.5, dice_weight=0.5, gamma=2.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.focal = FocalLoss(gamma=gamma, weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)

        return (
            self.ce_weight * ce_loss
            + self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
        )
