import json
import torch


def load_class_weights(path, device="cpu"):
    """
    Load class weights from a JSON file and return a torch tensor.
    """
    with open(path, "r") as f:
        weights = json.load(f)

    # Ensure order by class index
    weights = [weights[str(i)] for i in range(len(weights))]

    return torch.tensor(weights, dtype=torch.float32, device=device)
