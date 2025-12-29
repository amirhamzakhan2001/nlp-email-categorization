# src/config/inference_config.py

from pathlib import Path
import torch

from src.common.paths import DATA_DIR, ARTIFACT_DIR

# ------------------------
# Data (mutable)
# ------------------------

NEW_MAIL_CSV_PATH = DATA_DIR / 'raw' / "gmail_new_mail_buffer.csv"

# ------------------------
# Artifacts (immutable)
# ------------------------

CLUSTERING_ARTIFACT_DIR = ARTIFACT_DIR / "clustering"
SUPERVISED_ARTIFACT_DIR = ARTIFACT_DIR / "supervised"

CLUSTER_TREE_PATH = CLUSTERING_ARTIFACT_DIR / "cluster_tree.json"
LEAF_LABELS_PATH = CLUSTERING_ARTIFACT_DIR / "leaf_labels.json"

MODEL_PATH = SUPERVISED_ARTIFACT_DIR / "mlp_classifier.pt"
LABEL_ENCODER_PATH = SUPERVISED_ARTIFACT_DIR / "label_encoder.pkl"
TRAINING_METADATA_PATH = SUPERVISED_ARTIFACT_DIR / "training_metadata.json"

# ------------------------
# Device
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")