# src/pipelines/full_pipeline.py

from src.config.pipeline_config import PIPELINE_MODE, PipelineMode
from src.utils.reset_artifacts import reset_ml_artifacts
from src.common.logging import get_logger

# -------- Fetch --------
from src.fetch_gmail.fetch_and_clean_pipeline import main as fetch_full

# -------- Embedding --------
from src.embedding.embed_full import main as embed_full

# -------- Clustering --------
from src.clustering.cluster_full import main as run_clustering

# -------- Supervised --------
from src.supervised.train_supervised_full import main as train_supervised

# -------- Merge --------
from src.data_ops.csv_merge import merge_full_to_master

# -------- Visualization --------
from src.visualization.generate_visualizations import main as run_visualizations

from src.common.paths import DATA_DIR

logger = get_logger(__name__)


def main():
    logger.info("========== FULL PIPELINE STARTED ==========")
    logger.info("Pipeline mode = %s", PIPELINE_MODE)

    # 🔥 RETRAIN MODE HOOK (ONLY PLACE IT BELONGS)
    if PIPELINE_MODE == PipelineMode.RETRAIN:
        reset_ml_artifacts()

    # 1️⃣ Fetch all Gmail → gmail_cleaned.csv
    logger.info("Step 1/5 | Fetching full Gmail dataset")
    fetch_full()

    # 2️⃣ Embed + PCA → gmail_cluster_snapshot.csv
    logger.info("Step 2/5 | Embedding full dataset")
    embed_full()

    # 3️⃣ Unsupervised clustering + labeling
    logger.info("Step 3/5 | Running hierarchical clustering")
    run_clustering()

    # 4️⃣ Supervised training
    logger.info("Step 4/5 | Training supervised classifier")
    train_supervised()

    # 5️⃣ Merge cleaned + snapshot → gmail_master.csv
    logger.info("Step 5/5 | Merging into master CSV")

    merge_full_to_master(
        cleaned_csv=DATA_DIR / "raw" / "gmail_cleaned.csv",
        snapshot_csv=DATA_DIR / "processed" / "gmail_cluster_snapshot.csv",
        master_csv=DATA_DIR / "processed" / "gmail_master.csv",
    )

    # 6️⃣ Visualizations (NON-CRITICAL)
    logger.info("Step 6/6 | Generating visualizations")
    run_visualizations()

    logger.info("========== FULL PIPELINE COMPLETED ==========")


if __name__ == "__main__":
    main()