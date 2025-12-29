import os
import pandas as pd
from pathlib import Path
from src.common.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------
# FULL PIPELINE MERGE
# --------------------------------------------------

def merge_full_to_master(
    cleaned_csv: Path,
    snapshot_csv: Path,
    master_csv: Path,
    delete_snapshot: bool = True
):
    """
    Merge cleaned + cluster snapshot into gmail_master.csv
    (one-time full pipeline merge)
    """

    logger.info("Starting full CSV merge")

    df_clean = pd.read_csv(cleaned_csv)
    df_snap = pd.read_csv(snapshot_csv)

    if "Message_ID" not in df_clean or "Message_ID" not in df_snap:
        raise ValueError("Message_ID missing in one of the CSVs")

    # Inner join ensures consistency
    df_master = df_clean.merge(
        df_snap,
        on="Message_ID",
        how="inner",
        suffixes=("", "_ml")
    )

    master_csv.parent.mkdir(parents=True, exist_ok=True)

    df_master.to_csv(master_csv, index=False)
    logger.info(
        "Master CSV created | rows=%d cols=%d",
        df_master.shape[0],
        df_master.shape[1]
    )

    if delete_snapshot:
        snapshot_csv.unlink()
        logger.info("Deleted snapshot CSV")





# --------------------------------------------------
# INCREMENTAL MERGE
# --------------------------------------------------

def append_buffer_to_master(
    master_csv: Path,
    buffer_csv: Path,
    delete_buffer: bool = True
):
    """
    Append new inferred emails into gmail_master.csv
    """

    logger.info("Starting incremental merge")

    if not buffer_csv.exists():
        logger.warning("Buffer CSV not found, nothing to merge")
        return

    df_new = pd.read_csv(buffer_csv)

    if master_csv.exists():
        df_master = pd.read_csv(master_csv)

        combined = pd.concat(
            [df_master, df_new],
            ignore_index=True
        )

        # Safety dedupe
        combined = combined.drop_duplicates(
            subset=["Message_ID"],
            keep="last"
        )
    else:
        combined = df_new

    combined.to_csv(master_csv, index=False)

    logger.info(
        "Incremental merge completed | master_rows=%d",
        combined.shape[0]
    )

    if delete_buffer:
        buffer_csv.unlink()
        logger.info("Deleted buffer CSV")