# src/labeling/checkpoint_store.py

import pickle
from src.common.paths import ARTIFACT_DIR
from src.common.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_PATH = ARTIFACT_DIR / "clustering" / "checkpoint.pkl"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# LOAD CHECKPOINT (EXPLICIT)
# -------------------------

def load_leaf_checkpoint() -> dict:
    """
    Load checkpoint from disk.
    """
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        logger.info(
    "Loaded labeling checkpoint | entries=%d",
    len(data)
)
        return data

    logger.warning("No labeling checkpoint found — starting fresh")
    return {}


# -------------------------
# SAVE CHECKPOINT
# -------------------------

def save_leaf_checkpoint(
    leaf_checkpoint_dict: dict,
    cluster_id: str,
    label: str,
    summary: str,
):
    """
    Save partial labeling progress.
    Safe to call repeatedly.
    """
    if cluster_id in leaf_checkpoint_dict:
        return

    leaf_checkpoint_dict[cluster_id] = {
        "label": label,
        "summary": summary,
    }

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(leaf_checkpoint_dict, f)

    logger.debug("Checkpointed cluster | cluster_id=%s", cluster_id)