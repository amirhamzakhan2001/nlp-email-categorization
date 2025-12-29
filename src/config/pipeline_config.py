# src/config/pipeline_config.py

from enum import Enum
import os

class PipelineMode(str, Enum):
    FULL = "full"          # first ever run
    INCREMENTAL = "incremental"
    RETRAIN = "retrain"    # force full ML reset

# Read from env, default safe
PIPELINE_MODE = PipelineMode(
    os.getenv("PIPELINE_MODE", PipelineMode.INCREMENTAL)
)