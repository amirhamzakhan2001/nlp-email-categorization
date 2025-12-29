from pathlib import Path

from src.embedding.embedder import embed_texts
from src.embedding.pca_manager import apply_pca
from src.embedding.csv_io import load_csv, get_missing_embedding_indices, write_snapshot_embeddings
from src.config.embedding_config import CSV_BUFFER_PATH, TEXT_COLUMN, EMBEDDING_COLUMN
from src.common.logging import get_logger

logger = get_logger(__name__)

def main():
    df = load_csv(CSV_BUFFER_PATH)

    if EMBEDDING_COLUMN not in df.columns:
        df[EMBEDDING_COLUMN] = ""

    indices = get_missing_embedding_indices(df)

    if not indices:
        logger.warning("No new rows found for embedding")
        return

    texts = df.loc[indices, TEXT_COLUMN].fillna("").tolist()

    logger.info("Embedding incremental rows | count=%d", len(indices))

    emb_raw = embed_texts(texts)
    emb_final = apply_pca(emb_raw)

    write_snapshot_embeddings(df, indices, emb_final, CSV_BUFFER_PATH)
    logger.info(
        "Embedding completed | new_rows=%d",
        len(indices)
    )

if __name__ == "__main__":
    main()