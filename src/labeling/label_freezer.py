import json
from pathlib import Path
from typing import Dict

from src.common.paths import ARTIFACT_DIR
from src.common.logging import get_logger

logger = get_logger(__name__)

LEAF_LABELS_PATH = ARTIFACT_DIR  / "clustering" / "leaf_labels.json"

LEAF_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# PUBLIC API
# -------------------------

def freeze_leaf_labels(
    leaf_checkpoint_dict: Dict[str, Dict[str, str]]
):
    """
    Freeze final cluster labels & summaries into leaf_labels.json.
    This file is immutable once written.
    """

    if LEAF_LABELS_PATH.exists():
        raise RuntimeError(
            "leaf_labels.json already exists. "
            "Refreezing labels is not allowed."
        )

    with open(LEAF_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(leaf_checkpoint_dict, f, indent=2)

    logger.info(
    "Leaf labels frozen | path=%s",
    LEAF_LABELS_PATH
)