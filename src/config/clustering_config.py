from pathlib import Path
import os

# ------------------------
# BASE PATHS
# ------------------------
from src.common.paths import DATA_DIR, ARTIFACT_DIR

# ------------------------
# LIMITS (LLM + CLUSTERING)
# ------------------------
MAX_REQUESTS_PER_MINUTE = 12
WAIT_TIME_SECONDS = 60

MAX_DEPTH = 5
MIN_CLUSTER_SIZE = 10
MAX_LABEL_WORDS = 5
CHUNK_SIZE = 3200

MIN_SSE_GAIN = 0.08   # 8% improvement
N_BISECT_TRIALS = 6

# ------------------------
# CSV PATHS
# ------------------------
# 🚨 IMPORTANT:
# Clustering ALWAYS runs on SNAPSHOT, not cleaned CSV

DATASET_CSV_PATH = DATA_DIR / 'processed' / "gmail_cluster_snapshot.csv"