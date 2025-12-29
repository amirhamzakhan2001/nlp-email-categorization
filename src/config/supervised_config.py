import os
from pathlib import Path
import numpy as np
import torch

from src.common.paths import DATA_DIR, ARTIFACT_DIR


CSV_PATH = DATA_DIR / 'processed' / "gmail_cluster_snapshot.csv"

BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)