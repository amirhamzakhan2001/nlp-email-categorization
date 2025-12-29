# src/inference/label_path.py

import json
from pathlib import Path

from src.config.inference_config import (
    CLUSTER_TREE_PATH,
    LEAF_LABELS_PATH
)

# Load once (artifacts are immutable)
with open(CLUSTER_TREE_PATH) as f:
    CLUSTER_TREE = json.load(f)

with open(LEAF_LABELS_PATH) as f:
    LEAF_LABELS = json.load(f)


def build_label_path(cluster_id: str) -> str:
    """
    Build label path like:
    Root → Finance → Credit Card → OTP
    """

    labels = []
    cid = cluster_id

    while cid:
        if cid in LEAF_LABELS:
            labels.append(LEAF_LABELS[cid]["label"])
        cid = CLUSTER_TREE.get(cid, {}).get("parent")

    return " -> ".join(reversed(labels))