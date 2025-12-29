# src/embedding/model_loader.py

import json
from datetime import datetime

from models.embedding_models.qwen import QwenEmbedder
from src.common.paths import ARTIFACT_DIR
from src.config.embedding_config import (
    QWEN_MODEL_ID,
    QWEN_REVISION,
    HF_CACHE_DIR,
    MAX_LENGTH,
)
from src.common.logging import get_logger

logger = get_logger(__name__)

MODEL_INFO_PATH = ARTIFACT_DIR / "embeddings" / "qwen" / "model_info.json"
TOKENIZER_INFO_PATH = ARTIFACT_DIR / "embeddings" / "qwen" / "tokenizer_info.json"
MODEL_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)


def ensure_qwen_artifacts():
    """
    Instantiate embedder once and store immutable artifacts.
    """

    if MODEL_INFO_PATH.exists() and TOKENIZER_INFO_PATH.exists():
        return

    embedder = QwenEmbedder(
        model_id=QWEN_MODEL_ID,
        revision=QWEN_REVISION,
        cache_dir=HF_CACHE_DIR,
        max_length=MAX_LENGTH,
    )

    model = embedder.model
    tokenizer = embedder.tokenizer

    model_info = {
        "model_name": QWEN_MODEL_ID,
        "revision": QWEN_REVISION,
        "embedding_dimension": model.config.hidden_size,
        "pooling": "mean",
        "normalized": True,
        "torch_dtype": str(model.dtype),
        "device": embedder.device.type,
        "hf_cache_dir": str(HF_CACHE_DIR),
        "library": "transformers",
        "created_at": datetime.utcnow().isoformat(),
    }

    tokenizer_info = {
        "tokenizer_name": tokenizer.name_or_path,
        "max_length": MAX_LENGTH,
        "padding_side": tokenizer.padding_side,
        "truncation": True,
        "fast_tokenizer": tokenizer.is_fast,
        "created_at": datetime.utcnow().isoformat(),
    }

    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)

    with open(TOKENIZER_INFO_PATH, "w") as f:
        json.dump(tokenizer_info, f, indent=2)

    logger.info("Qwen embedding artifacts created")