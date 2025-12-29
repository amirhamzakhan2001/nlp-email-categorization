import json
from pathlib import Path
from typing import Dict, List

from src.common.paths import ARTIFACT_DIR
from src.common.logging import get_logger

logger = get_logger(__name__)

CLUSTER_TREE_PATH = ARTIFACT_DIR / "clustering" / "cluster_tree.json"

CLUSTER_TREE_PATH.parent.mkdir(parents=True, exist_ok=True)



# -------------------------
# PUBLIC API
# -------------------------

def build_and_save_cluster_tree(
    cluster_members: Dict[str, List[int]]
):
    """
    Build full cluster hierarchy from dot-encoded cluster_ids
    and save as cluster_tree.json
    """

    tree = {}

    # First pass: create nodes
    for cid, members in cluster_members.items():
        parent = cid.rsplit(".", 1)[0] if "." in cid else None

        tree[cid] = {
            "cluster_id": cid,
            "parent": parent,
            "children": [],
            "depth": cid.count("."),
            "n_samples": len(members)
        }

    # Second pass: populate children
    for cid, node in tree.items():
        parent = node["parent"]
        if parent and parent in tree:
            tree[parent]["children"].append(cid)

    # Save immutable artifact
    with open(CLUSTER_TREE_PATH, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2)

    logger.info(
    "Cluster tree hierarchy saved | path=%s",
    CLUSTER_TREE_PATH
)