# src/embedding/embedder.py

import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

from models.embedding_models.qwen import QwenEmbedder
from src.embedding.model_loader import ensure_qwen_artifacts
from src.config.embedding_config import (
    QWEN_MODEL_ID,
    QWEN_REVISION,
    HF_CACHE_DIR,
    MAX_LENGTH,
    BATCH_SIZE,
)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed texts using Qwen and return L2-normalized embeddings.
    """

    # Ensure artifacts exist (write-once)
    ensure_qwen_artifacts()

    embedder = QwenEmbedder(
        model_id=QWEN_MODEL_ID,
        revision=QWEN_REVISION,
        cache_dir=HF_CACHE_DIR,
        max_length=MAX_LENGTH,
    )

    vectors = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        batch_vecs = embedder.embed(batch, batch_size=len(batch))
        vectors.append(batch_vecs)

    X = np.vstack(vectors)
    return normalize(X, axis=1)