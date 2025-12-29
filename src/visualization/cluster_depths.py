import json
import matplotlib.pyplot as plt

from src.outputs.visualization_utils import save_figure


def plot_cluster_depths(cluster_tree_path):
    """
    Histogram of cluster depths
    """
    with open(cluster_tree_path) as f:
        tree = json.load(f)

    depths = [node["depth"] for node in tree.values()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(depths, bins=range(max(depths) + 2), edgecolor="black")

    ax.set_title("Cluster Depth Distribution")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Number of Clusters")

    save_figure(fig, "cluster_depth_distribution")
    plt.close(fig)