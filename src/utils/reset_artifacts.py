# src/utils/reset_artifacts.py

import shutil
from pathlib import Path
from src.common.paths import ARTIFACT_DIR
from src.common.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_SUBDIRS = [
    ARTIFACT_DIR / "pca",
    ARTIFACT_DIR / "supervised",
    ARTIFACT_DIR / "clustering",
]

def reset_ml_artifacts():
    """
    Deletes ALL ML-generated artifacts.
    Safe to call multiple times.
    """
    logger.warning("ML RETRAIN MODE ENABLED — resetting artifacts")

    for path in ARTIFACT_SUBDIRS:
        if path.exists():
            shutil.rmtree(path)
            logger.info("Deleted artifact directory: %s", path)
        else:
            logger.info("Artifact directory not found (skipped): %s", path)