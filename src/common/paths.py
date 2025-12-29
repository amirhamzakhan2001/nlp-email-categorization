# src/common/paths.py

from pathlib import Path

# Project root (one and only one definition)
BASE_DIR = Path(__file__).resolve().parents[2]

# Core directories
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Subdirectories (optional)
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure existence (safe)
for p in [
    DATA_DIR,
    ARTIFACT_DIR,
    OUTPUTS_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)