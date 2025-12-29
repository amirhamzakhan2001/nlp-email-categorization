import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.outputs.visualization_utils import save_figure


def plot_embedding_map(
    csv_path,
    embedding_column="body_embeddings",
    label_column="cluster_id",
    max_points=5000,
):
    """
    2D scatter plot of embeddings (PCA already applied).
    """
    df = pd.read_csv(csv_path)

    # Parse embeddings
    embeddings = df[embedding_column].apply(eval).to_list()
    X = np.array(embeddings)

    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X = X[idx]
        labels = df[label_column].iloc[idx]
    else:
        labels = df[label_column]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels.astype("category").cat.codes,
        s=5,
        alpha=0.6
    )

    ax.set_title("Embedding Map (PCA space)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    save_figure(fig, "embedding_map")
    plt.close(fig)