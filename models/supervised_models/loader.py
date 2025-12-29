import torch
from pathlib import Path
from .mlp import MLPClassifier


def load_mlp_from_artifact(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)

    architecture = checkpoint.get("architecture")
    if architecture is None:
        raise RuntimeError("Model architecture missing in checkpoint")

    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["n_classes"],
        hidden_layers=architecture["hidden_layers"],
        dropout=architecture["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model