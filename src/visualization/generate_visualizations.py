from src.common.paths import DATA_DIR, ARTIFACT_DIR
from src.visualization.cluster_sizes import plot_cluster_sizes
from src.visualization.embedding_map import plot_embedding_map
from src.visualization.cluster_depths import plot_cluster_depths
from src.visualization.sse_metrics import plot_sse_gain


def main():
    master_csv = DATA_DIR / "processed" / "gmail_master.csv"
    cluster_tree = ARTIFACT_DIR / "clustering" / "cluster_tree.json"
    sse_metrics = ARTIFACT_DIR / "outputs" / "metrics" / "bisecting_kmeans_metrics.json"

    plot_cluster_sizes(master_csv)
    plot_embedding_map(master_csv)
    plot_cluster_depths(cluster_tree)
    plot_sse_gain(sse_metrics)


if __name__ == "__main__":
    main()