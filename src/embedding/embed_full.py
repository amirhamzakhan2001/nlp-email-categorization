from pathlib import Path

from src.embedding.embedder import embed_texts
from src.embedding.pca_manager import fit_and_save_pca
from src.embedding.csv_io import load_csv, init_snapshot_df, write_snapshot_embeddings
from src.config.embedding_config import (
    CSV_MASTER,
    CSV_SNAPSHOT_PATH,
    TEXT_COLUMN
)
from src.common.logging import get_logger

logger = get_logger(__name__)

def main():
    # Load cleaned emails (INPUT ONLY)
 
    cleaned_df = load_csv(CSV_MASTER)

    texts = cleaned_df[TEXT_COLUMN].fillna("").tolist()

    # Embed + PCA
    emb_raw = embed_texts(texts)
    emb_final = fit_and_save_pca(emb_raw)

    # Create snapshot + save embeddings
    snapshot_df = init_snapshot_df(cleaned_df)

    write_snapshot_embeddings(
        snapshot_df,
        snapshot_df.index.tolist(),
        emb_final,
        CSV_SNAPSHOT_PATH
    )

    logger.info("Cluster snapshot created with PCA embeddings")


if __name__ == "__main__":
    main()