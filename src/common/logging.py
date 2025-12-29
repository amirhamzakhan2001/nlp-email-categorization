# src/common/logging.py

import logging
from src.common.paths import OUTPUTS_DIR

LOG_DIR = OUTPUTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = (
    "[%(asctime)s] "
    "[%(levelname)s] "
    "[%(name)s] "
    "%(message)s"
)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Prevent duplicate handlers

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT))

    # File handler
    fh = logging.FileHandler(LOG_DIR / "app.log")
    fh.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger