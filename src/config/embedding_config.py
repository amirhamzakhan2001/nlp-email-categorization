from pathlib import Path

# ------------------------
# BASE PATHS
# ------------------------
from src.common.paths import DATA_DIR, BASE_DIR, ARTIFACT_DIR, MODELS_DIR
HF_CACHE_DIR = BASE_DIR / "hf_models"


# ------------------------
# CSV CONFIG
# ------------------------
CSV_MASTER = DATA_DIR / 'raw' / "gmail_cleaned.csv"
CSV_BUFFER_PATH = DATA_DIR / 'raw' / "gmail_new_mail_buffer.csv"

# NEW: cluster snapshot (ML-owned)
CSV_SNAPSHOT_PATH = DATA_DIR / 'processed' / "gmail_cluster_snapshot.csv"


TEXT_COLUMN = "Cleaned_body"
EMBEDDING_COLUMN = "body_embeddings"

# ------------------------ Areas around code to edit
# MODEL CONFIG
# ------------------------
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
QWEN_REVISION = "main"

BATCH_SIZE = 8
MAX_LENGTH = 8192

# ------------------------
# PCA CONFIG
# ------------------------
PCA_DIM = 256
PCA_RANDOM_SEED = 42