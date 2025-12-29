import os
import json
import tempfile
import csv
import pandas as pd
from pathlib import Path

from src.config.embedding_config import EMBEDDING_COLUMN


def load_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str)

def get_missing_embedding_indices(df: pd.DataFrame):
    if EMBEDDING_COLUMN not in df.columns:
        df[EMBEDDING_COLUMN] = ""
    mask = df[EMBEDDING_COLUMN].isna() | (df[EMBEDDING_COLUMN].str.strip() == "")
    return df.index[mask].tolist()


def init_snapshot_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML snapshot from cleaned CSV.
    Only keeps columns required for ML.
    """
    snapshot_df = pd.DataFrame()
    snapshot_df["Message_ID"] = source_df["Message_ID"]
    snapshot_df["Cleaned_body"] = source_df["Cleaned_body"]
    snapshot_df[EMBEDDING_COLUMN] = ""

    return snapshot_df


def write_snapshot_embeddings(
    snapshot_df: pd.DataFrame,
    indices,
    embeddings,
    snapshot_path: Path
):
    for idx, vec in zip(indices, embeddings):
        snapshot_df.at[idx, EMBEDDING_COLUMN] = json.dumps(vec.tolist())

    fd, tmp_file = tempfile.mkstemp(suffix=".tmp")
    os.close(fd)

    snapshot_df.to_csv(tmp_file, index=False, quoting=csv.QUOTE_ALL)
    os.replace(tmp_file, snapshot_path)