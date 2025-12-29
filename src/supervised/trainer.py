import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.common.logging import get_logger

logger = get_logger(__name__)


def train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    device,
    batch_size,
    max_epochs,
    patience,
    lr,
    weight_decay
):
    # -------------------------
    # Class weights (robust)
    # -------------------------
    n_classes = int(np.max(y_train)) + 1
    class_counts = np.bincount(y_train, minlength=n_classes)

    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # -------------------------
    # Data loaders
    # -------------------------
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    # -------------------------
    # Training loop
    # -------------------------
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_val_acc = 0.0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            val_loss = criterion(logits, y_val_t).item()
            preds = torch.argmax(logits, dim=1)
            val_acc = (preds == y_val_t).float().mean().item()

        logger.debug(
            "Epoch=%02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_acc
        )

        # -------------------------
        # Early stopping
        # -------------------------
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.warning(
                    "Early stopping triggered | epoch=%d",
                    epoch
                )
                break

    # -------------------------
    # Restore best model
    # -------------------------
    model.load_state_dict(best_state)

    return model, best_epoch, best_val_acc