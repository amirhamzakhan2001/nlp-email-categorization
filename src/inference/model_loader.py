import json
import joblib
from pathlib import Path

from models.supervised_models.loader import load_mlp_from_artifact
from src.config.supervised_config import DEVICE
from src.common.paths import ARTIFACT_DIR


# -------------------------
# Artifact paths
# -------------------------

SUPERVISED_DIR = ARTIFACT_DIR / "supervised"

MODEL_PATH = SUPERVISED_DIR / "mlp_classifier.pt"
LABEL_ENCODER_PATH = SUPERVISED_DIR / "label_encoder.pkl"
METADATA_PATH = SUPERVISED_DIR / "training_metadata.json"


# -------------------------
# Loader
# -------------------------

def load_supervised_model(device=DEVICE):
    """
    Load supervised MLP classifier and label encoder for inference.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")

    # ---- Load metadata (sanity checks) ----
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    # ---- Load model checkpoint ----
    model = load_mlp_from_artifact(MODEL_PATH, device)

    # ---- Load label encoder ----
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return model, label_encoder, metadata