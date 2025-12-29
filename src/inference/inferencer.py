# src/inference/inferencer.py
import json
import numpy as np
import torch
from sklearn.preprocessing import normalize

from src.inference.label_path import build_label_path
from src.config.inference_config import LEAF_LABELS_PATH

# Load labels once
with open(LEAF_LABELS_PATH) as f:
    LEAF_LABELS = json.load(f)


def run_inference(
    df,
    new_indices,
    model,
    label_encoder,
    device
):
    X_new = np.vstack(df.loc[new_indices, "body_embeddings"].values)
    X_new = normalize(X_new, norm="l2")

    X_new_t = torch.tensor(X_new, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_new_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    cluster_ids = label_encoder.inverse_transform(preds)

    for idx, cid in zip(new_indices, cluster_ids):
        df.at[idx, "cluster_id"] = cid
        df.at[idx, "cluster_leaf_label"] = LEAF_LABELS[cid]["label"]
        df.at[idx, "cluster_leaf_summary"] = LEAF_LABELS[cid]["summary"]
        df.at[idx, "cluster_path"] = build_label_path(cid)

    return df