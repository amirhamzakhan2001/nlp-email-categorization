# src/clustering/kmeans_recursive.py

import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

from src.config.clustering_config import (
    MAX_DEPTH,
    MIN_CLUSTER_SIZE,
    MIN_SSE_GAIN,
    N_BISECT_TRIALS
)

from src.labeling.label_generator import generate_label_and_summary, generate_parent_label_and_summary
from src.labeling.checkpoint_store import (
    load_leaf_checkpoint,
    save_leaf_checkpoint
)
from src.common.paths import ARTIFACT_DIR

# -----------------------------
# Global stores
# -----------------------------

cluster_members = {}
sse_metrics = {}
leaf_checkpoint_dict = load_leaf_checkpoint()

METRICS_PATH = ARTIFACT_DIR / "outputs" / "metrics" / "bisecting_kmeans_metrics.json"
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def compute_sse(X):
    centroid = X.mean(axis=0)
    return float(np.sum((X - centroid) ** 2))


def best_bisecting_split(X, n_trials):
    best_labels = None
    best_inertia = float("inf")

    for seed in range(n_trials):
        kmeans = KMeans(
            n_clusters=2,
            random_state=seed,
            n_init=1
        )
        labels = kmeans.fit_predict(X)

        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = labels

    return best_labels, best_inertia


# -----------------------------
# Recursive bisecting k-means
# -----------------------------

def recursive_clustering(X, texts, indices, cluster_id, depth=0):
    cluster_members.setdefault(cluster_id, indices.tolist())

    # ---- stopping: depth or size ----
    if depth >= MAX_DEPTH or len(indices) <= MIN_CLUSTER_SIZE:
        if cluster_id not in leaf_checkpoint_dict:
            label, summary = generate_label_and_summary(texts, X)
            save_leaf_checkpoint(leaf_checkpoint_dict, cluster_id, label, summary)
        return

    parent_sse = compute_sse(X)

    # ---- bisect ----
    labels, children_sse = best_bisecting_split(X, N_BISECT_TRIALS)

    # Degenerate split
    if labels is None or len(set(labels)) < 2:
        if cluster_id not in leaf_checkpoint_dict:
            label, summary = generate_label_and_summary(texts, X)
            save_leaf_checkpoint(leaf_checkpoint_dict, cluster_id, label, summary)
        return

    sse_gain = (parent_sse - children_sse) / max(parent_sse, 1e-9)

    # ---- stopping: poor gain ----
    if sse_gain < MIN_SSE_GAIN:
        if cluster_id not in leaf_checkpoint_dict:
            label, summary = generate_label_and_summary(texts, X)
            save_leaf_checkpoint(cluster_id, label, summary)
        return

    # ---- record metrics ----
    sse_metrics[cluster_id] = {
        "depth": depth,
        "n_samples": len(indices),
        "parent_sse": parent_sse,
        "children_sse": children_sse,
        "sse_gain": sse_gain
    }

    # ---- recurse into children ----
    for i in [0, 1]:
        mask = labels == i
        child_id = f"{cluster_id}.{i+1}"

        sub_X = X[mask]
        sub_texts = [texts[j] for j, m in enumerate(mask) if m]
        sub_indices = indices[mask]

        recursive_clustering(
            sub_X,
            sub_texts,
            sub_indices,
            child_id,
            depth + 1
        )

    # ---- parent labeling (merge children) ----
    if cluster_id not in leaf_checkpoint_dict:
        child_labels = []
        child_summaries = []

        for cid in cluster_members:
            if cid.startswith(cluster_id + ".") and cid.count(".") == cluster_id.count(".") + 1:
                if cid in leaf_checkpoint_dict:
                    child_labels.append(leaf_checkpoint_dict[cid]["label"])
                    child_summaries.append(leaf_checkpoint_dict[cid]["summary"])

        if child_labels:
            label, summary = generate_parent_label_and_summary(child_labels, child_summaries)
            save_leaf_checkpoint(leaf_checkpoint_dict, cluster_id, label, summary)


def save_sse_metrics():
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(sse_metrics, f, indent=2)