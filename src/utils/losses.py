import json
import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255


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
    def __init__(self, gamma=2.0, weight=None, reduction="mean", ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.new_tensor(0.0)

        logits = logits.permute(0, 2, 3, 1)[valid_mask]
        target = target[valid_mask]

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
    def __init__(self, smooth=1.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.new_tensor(0.0)

        probs = F.softmax(logits, dim=1)
        probs = probs.permute(0, 2, 3, 1)[valid_mask]
        target = target[valid_mask]

        target_oh = F.one_hot(target, num_classes=num_classes).float()

        intersection = torch.sum(probs * target_oh, dim=0)
        cardinality = torch.sum(probs + target_oh, dim=0)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    CE + Focal + Dice loss.
    Useful for highly imbalanced semantic segmentation datasets.
    """

    def __init__(
        self,
        class_weights=None,
        ce_weight=1.0,
        focal_weight=0.5,
        dice_weight=0.5,
        gamma=2.0,
        ignore_index=IGNORE_INDEX,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.focal = FocalLoss(gamma=gamma, weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)

        return (
            self.ce_weight * ce_loss
            + self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
        )
