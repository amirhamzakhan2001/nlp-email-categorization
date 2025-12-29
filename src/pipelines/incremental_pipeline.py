# src/pipelines/incremental_pipeline.py

from src.common.logging import get_logger

# -------- Fetch --------
from src.fetch_gmail.fetch_latest_pipeline import main as fetch_incremental

# -------- Embedding --------
from src.embedding.embed_incremental import main as embed_incremental

# -------- Inference --------
from src.inference.infer_supervised import main as run_inference

# -------- Merge --------
from src.data_ops.csv_merge import append_buffer_to_master

from src.common.paths import DATA_DIR

logger = get_logger(__name__)


def main():
    logger.info("========== INCREMENTAL PIPELINE STARTED ==========")

    # 1️⃣ Fetch new Gmail → gmail_new_mail_buffer.csv
    logger.info("Step 1/4 | Fetching incremental emails")
    fetch_incremental()

    # 2️⃣ Embed new emails (PCA reuse)
    logger.info("Step 2/4 | Embedding new emails")
    embed_incremental()

    # 3️⃣ Supervised inference
    logger.info("Step 3/4 | Running supervised inference")
    run_inference()

    # 4️⃣ Append buffer → master, delete buffer
    logger.info("Step 4/4 | Appending to master CSV")

    append_buffer_to_master(
        master_csv=DATA_DIR / "processed" / "gmail_master.csv",
        buffer_csv=DATA_DIR / "raw" / "gmail_new_mail_buffer.csv",
    )

    logger.info("========== INCREMENTAL PIPELINE COMPLETED ==========")


if __name__ == "__main__":
    main()