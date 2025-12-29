import json
import matplotlib.pyplot as plt

from src.outputs.visualization_utils import save_figure


def plot_sse_gain(metrics_path):
    """
    SSE gain vs depth
    """
    with open(metrics_path) as f:
        metrics = json.load(f)

    depths = []
    gains = []

    for v in metrics.values():
        depths.append(v["depth"])
        gains.append(v["sse_gain"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(depths, gains, alpha=0.7)

    ax.set_xlabel("Cluster Depth")
    ax.set_ylabel("SSE Gain")
    ax.set_title("SSE Gain vs Cluster Depth")

    save_figure(fig, "sse_gain_vs_depth")
    plt.close(fig)