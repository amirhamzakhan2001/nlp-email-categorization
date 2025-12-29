import json
import joblib
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from src.common.paths import ARTIFACT_DIR
from src.config.embedding_config import (
    PCA_DIM,
    PCA_RANDOM_SEED
)
from src.common.logging import get_logger

logger = get_logger(__name__)

# ----------------------------
# ARTIFACT PATHS (OWNED HERE)
# ----------------------------

PCA_MODEL_PATH = ARTIFACT_DIR / "pca" / f"pca_{PCA_DIM}.joblib"
PCA_METADATA_PATH = ARTIFACT_DIR / "pca" / "pca_metadata.json"


# ----------------------------
# UTILITIES
# ----------------------------

def _hash_embeddings(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return f"sha256:{h.hexdigest()}"


# ----------------------------
# FIT (ONCE ONLY)
# ----------------------------

def fit_and_save_pca(embeddings: np.ndarray) -> np.ndarray:
    """
    Fit PCA on full embeddings ONCE, save PCA + metadata,
    and return normalized PCA embeddings.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")


    input_dim = embeddings.shape[1]
    logger.info(
        "Fitting PCA | input_dim=%d output_dim=%d",
        input_dim,
        PCA_DIM
    )

    pca = PCA(
        n_components=PCA_DIM,
        random_state=PCA_RANDOM_SEED
    )

    X_pca = pca.fit_transform(embeddings)
    X_pca = normalize(X_pca, axis=1)

    # Save PCA model
    joblib.dump(pca, PCA_MODEL_PATH)

    # Save metadata
    metadata = {
        "input_dimension": input_dim,
        "output_dimension": PCA_DIM,
        "explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "n_samples_fit": embeddings.shape[0],
        "random_seed": PCA_RANDOM_SEED,
        "normalized": True,
        "embedding_hash": _hash_embeddings(embeddings),
        "fit_timestamp": datetime.utcnow().isoformat()
    }

    with open(PCA_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("PCA model saved | path=%s", PCA_MODEL_PATH)
    logger.info("PCA metadata saved | path=%s", PCA_METADATA_PATH)

    return X_pca


# ----------------------------
# APPLY (INCREMENTAL SAFE)
# ----------------------------

def apply_pca(embeddings: np.ndarray) -> np.ndarray:
    """
    Apply previously fitted PCA to new embeddings.
    Never refits PCA.
    """
    if not PCA_MODEL_PATH.exists():
        raise RuntimeError("PCA model not found. Run full embedding first.")

    if not PCA_METADATA_PATH.exists():
        raise RuntimeError("PCA metadata missing.")

    with open(PCA_METADATA_PATH) as f:
        meta = json.load(f)

    if embeddings.shape[1] != meta["input_dimension"]:
        raise ValueError(
            f"Embedding dim mismatch: "
            f"{embeddings.shape[1]} != {meta['input_dimension']}"
        )

    pca = joblib.load(PCA_MODEL_PATH)
    X_pca = pca.transform(embeddings)

    return normalize(X_pca, axis=1)