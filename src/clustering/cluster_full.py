# src/pipelines/run_unsupervised_clustering.py

import os
import tempfile

from src.config.clustering_config import DATASET_CSV_PATH
from src.clustering.data_loader import load_embeddings_and_texts
from src.clustering.kmeans_recursive import (
    recursive_clustering,
    cluster_members
)

from src.graph.label_path import build_label_path
from src.labeling.checkpoint_store import load_leaf_checkpoint
from src.clustering.cluster_tree_builder import build_and_save_cluster_tree
from src.labeling.label_freezer import freeze_leaf_labels
from src.common.paths import ARTIFACT_DIR
from src.common.logging import get_logger
from src.outputs.metrics_writer import save_metrics
from src.clustering.kmeans_recursive import sse_metrics

logger = get_logger(__name__)


def main():

    leaf_checkpoint_dict = load_leaf_checkpoint()
    # -------------------------
    # 1. Load dataset
    # -------------------------
    df, X, texts, indices = load_embeddings_and_texts(DATASET_CSV_PATH)

    # -------------------------
    # 2. Run hierarchical clustering
    # -------------------------
    recursive_clustering(
        X=X,
        texts=texts,
        indices=indices,
        cluster_id="root",
        depth=0
    )

    # -------------------------
    # 3. Save cluster hierarchy (IMMUTABLE)
    # -------------------------
    build_and_save_cluster_tree(cluster_members)

    # -------------------------
    # 4. Freeze labels (ONLY ONCE)
    # -------------------------
    leaf_labels_path = ARTIFACT_DIR / "clustering" / "leaf_labels.json"
    if not leaf_labels_path.exists():
        freeze_leaf_labels(leaf_checkpoint_dict)
    else:
        logger.warning("leaf_labels.json already exists — skipping freeze step")

    # -------------------------
    # 5. Assign deepest (leaf) cluster per email
    # -------------------------
    cluster_id_per_email = [None] * len(df)

    for cid, members in cluster_members.items():
        for idx in members:
            if (
                cluster_id_per_email[idx] is None
                or cid.count(".") > cluster_id_per_email[idx].count(".")
            ):
                cluster_id_per_email[idx] = cid

    # -------------------------
    # 6. Populate dataset CSV (DERIVED VIEW)
    # -------------------------
    df["cluster_id"] = cluster_id_per_email
    df["cluster_path"] = [
        build_label_path(cid, leaf_checkpoint_dict) for cid in cluster_id_per_email
    ]

    df["cluster_leaf_label"] = [
        leaf_checkpoint_dict.get(cid, {}).get("label", "")
        for cid in cluster_id_per_email
    ]

    df["cluster_leaf_summary"] = [
        leaf_checkpoint_dict.get(cid, {}).get("summary", "")
        for cid in cluster_id_per_email
    ]

    # -------------------------
    # 7. Atomic CSV write
    # -------------------------
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=os.path.dirname(DATASET_CSV_PATH),
        suffix=".tmp",
        encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp.name, index=False)

    os.replace(tmp.name, DATASET_CSV_PATH)

    logger.info("Unsupervised clustering completed successfully")


    save_metrics("bisecting_kmeans", sse_metrics)


if __name__ == "__main__":
    main()