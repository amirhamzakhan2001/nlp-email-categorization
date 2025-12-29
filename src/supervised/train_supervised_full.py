import json
import joblib
import torch
from datetime import datetime

import mlflow
import mlflow.pytorch

from src.common.paths import ARTIFACT_DIR
from src.config.supervised_config import *
from src.supervised.sup_data_loader import load_csv_embeddings
from src.supervised.dataset_builder import build_datasets
from src.supervised.evaluator import evaluate
from src.supervised.automl import run_mlp_automl
from src.common.logging import get_logger

logger = get_logger(__name__)

# -------------------------
# Artifact paths
# -------------------------
SUP_ARTIFACT_DIR = ARTIFACT_DIR / "supervised"
SUP_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SUP_ARTIFACT_DIR / "mlp_classifier.pt"
LABEL_ENCODER_PATH = SUP_ARTIFACT_DIR / "label_encoder.pkl"
METADATA_PATH = SUP_ARTIFACT_DIR / "training_metadata.json"


def main():
    logger.info("Supervised training started")

    # -------------------------
    # MLflow setup
    # -------------------------
    mlflow.set_experiment("gmail_email_clustering_supervised")

    with mlflow.start_run(run_name="mlp_automl_training"):

        # -------------------------
        # Load data
        # -------------------------
        X, y, _ = load_csv_embeddings(CSV_PATH)

        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            label_encoder, n_classes
        ) = build_datasets(
            X, y, TEST_SIZE, VAL_SIZE, RANDOM_SEED
        )

        logger.info(
            "Dataset loaded | samples=%d features=%d classes=%d",
            X.shape[0],
            X.shape[1],
            n_classes
        )

        # -------------------------
        # Log global params
        # -------------------------
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "random_seed": RANDOM_SEED,
        })

        # -------------------------
        # AutoML
        # -------------------------
        best_model, best_config, best_metrics = run_mlp_automl(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            input_dim=X_train.shape[1],
            n_classes=n_classes,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        test_acc = evaluate(best_model, X_test, y_test, DEVICE)
        # -------------------------
        # Log metrics
        # -------------------------
        mlflow.log_metrics({
            "best_val_accuracy": best_metrics["val_accuracy"],
            "test_accuracy": test_acc,
            "best_epoch": best_metrics["best_epoch"],
        })

        # -------------------------
        # Save model artifact
        # -------------------------
        torch.save(
            {
                "model_state_dict": best_model.state_dict(),
                "input_dim": X_train.shape[1],
                "n_classes": n_classes,
                "architecture": best_config,
            },
            MODEL_PATH
        )

        joblib.dump(label_encoder, LABEL_ENCODER_PATH)

        # -------------------------
        # Save metadata
        # -------------------------
        metadata = {
            "model_type": "MLPClassifier",
            "input_dim": X_train.shape[1],
            "n_classes": n_classes,
            "architecture": best_config,
            "best_epoch": best_metrics["best_epoch"],
            "best_val_accuracy": best_metrics["val_accuracy"],
            "test_accuracy": best_metrics["test_accuracy"],
            "dataset": str(CSV_PATH),
            "trained_at": datetime.utcnow().isoformat(),
        }

        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        # -------------------------
        # MLflow artifacts
        # -------------------------
        mlflow.log_artifact(METADATA_PATH)
        mlflow.log_artifact(LABEL_ENCODER_PATH)
        mlflow.pytorch.log_model(best_model, artifact_path="model")

        logger.info("Supervised training completed successfully")


if __name__ == "__main__":
    main()