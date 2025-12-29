# src/supervised/automl.py

from typing import List, Dict, Tuple
import torch

from src.supervised.trainer import train_model
from src.supervised.evaluator import evaluate
from models.supervised_models.mlp import MLPClassifier
from src.common.logging import get_logger

logger = get_logger(__name__)


# -------------------------
# Search space
# -------------------------

MLP_SEARCH_SPACE: List[Dict] = [
    {"hidden_layers": [512, 256], "dropout": 0.2},
    {"hidden_layers": [768, 384], "dropout": 0.3},
    {"hidden_layers": [512, 256, 128], "dropout": 0.25},
    {"hidden_layers": [1024, 512], "dropout": 0.35},
    {"hidden_layers": [580, 320 ], "dropout": 0.25},
]


# -------------------------
# AutoML runner
# -------------------------

def run_mlp_automl(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    input_dim: int,
    n_classes: int,
    device,
    batch_size,
    max_epochs,
    patience,
    lr,
    weight_decay,
) -> Tuple[torch.nn.Module, dict, dict]:
    """
    Runs AutoML over predefined MLP architectures.
    Returns:
        best_model
        best_config
        best_metrics
    """

    best_val_acc = -1.0
    best_model = None
    best_config = None
    best_metrics = None

    logger.info(
        "AutoML started | trials=%d",
        len(MLP_SEARCH_SPACE)
    )

    for idx, config in enumerate(MLP_SEARCH_SPACE, start=1):
        logger.info(
            "AutoML trial %d/%d | layers=%s dropout=%.2f",
            idx,
            len(MLP_SEARCH_SPACE),
            config["hidden_layers"],
            config["dropout"]
        )

        model = MLPClassifier(
            input_dim=input_dim,
            num_classes=n_classes,
            hidden_layers=config["hidden_layers"],
            dropout=config["dropout"],
        ).to(device)

        model, best_epoch, val_acc = train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            device,
            batch_size,
            max_epochs,
            patience,
            lr,
            weight_decay
        )


        logger.info(
            "Trial result | val_acc=%.4f  epoch=%d",
            val_acc,
            best_epoch
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_config = config
            best_metrics = {
                "best_epoch": best_epoch,
                "val_accuracy": val_acc,
            }

            logger.info("✅ New best model selected")

    logger.info(
        "AutoML completed | best_val_acc=%.4f",
        best_val_acc
    )

    return best_model, best_config, best_metrics