# src/inference/infer_supervised.py

import os
import tempfile

from src.config.inference_config import (
    NEW_MAIL_CSV_PATH,
    DEVICE
)

from src.inference.inf_data_loader import load_dataset
from src.inference.model_loader import load_supervised_model
from src.inference.inferencer import run_inference
from src.common.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Inference device selected | device=%s", DEVICE)

    df, new_indices = load_dataset(NEW_MAIL_CSV_PATH)

    if not new_indices:
        logger.warning("No new emails found — skipping inference")
        return

    logger.info(
        "New emails detected for inference | count=%d",
        len(new_indices)
    )

    model, label_encoder, _ = load_supervised_model()
    logger.info("Supervised inference model loaded")

    df = run_inference(
        df=df,
        new_indices=new_indices,
        model=model,
        label_encoder=label_encoder,
        device=DEVICE
    )

    # Atomic write
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=os.path.dirname(NEW_MAIL_CSV_PATH),
        suffix=".tmp",
        encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp.name, index=False)

    os.replace(tmp.name, NEW_MAIL_CSV_PATH)

    logger.info("Supervised cluster inference completed successfully")


if __name__ == "__main__":
    main()