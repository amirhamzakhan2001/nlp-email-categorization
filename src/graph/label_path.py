# src/graph/label_path.py

def build_label_path(cluster_id: str, leaf_checkpoint: dict) -> str:
    if not cluster_id:
        return ""

    parts = cluster_id.split(".")
    ids = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]

    labels = [
        leaf_checkpoint[i]["label"]
        for i in ids
        if i in leaf_checkpoint
    ]

    return " -> ".join(labels)