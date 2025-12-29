import pandas as pd
import matplotlib.pyplot as plt

from src.outputs.visualization_utils import save_figure


def plot_cluster_sizes(csv_path):
    """
    Bar plot: number of emails per cluster
    """
    df = pd.read_csv(csv_path)

    sizes = (
        df["cluster_id"]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sizes.plot(kind="bar", ax=ax)

    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Emails")

    save_figure(fig, "cluster_size_distribution")
    plt.close(fig)