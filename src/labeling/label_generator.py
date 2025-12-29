# src/labeling/label_generator.py

import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config.clustering_config import CHUNK_SIZE
from src.labeling.prompt_builder import label_prompting
from src.common.logging import get_logger

logger = get_logger(__name__)


# -------------------------
# Utilities
# -------------------------

def embedding_dedupe(texts, embeddings, threshold):
    """
    Remove near-duplicate emails using embedding cosine similarity.
    """
    if len(texts) <= 1:
        return texts

    embeddings = np.array(embeddings)
    sims = cosine_similarity(embeddings)

    taken = set()
    reps = []

    for i in range(len(texts)):
        if i not in taken:
            reps.append(i)
            taken.update(np.where(sims[i] >= threshold)[0])

    return [texts[i] for i in reps]


# -------------------------
# Leaf cluster labeling
# -------------------------

def generate_label_and_summary(
    texts,
    embeddings,
    max_emails_per_batch=10
):
    """
    Generate label and summary for a LEAF cluster using email texts.
    Pure function: no side effects.
    """

    if not texts:
        return "Empty cluster", ""

    # Deduplicate representative emails
    texts = embedding_dedupe(texts, embeddings, threshold=0.95)

    # Batch emails
    email_batches = [
        texts[i:i + max_emails_per_batch]
        for i in range(0, len(texts), max_emails_per_batch)
    ]

    batch_labels = []
    batch_summaries = []

    for batch_idx, batch in enumerate(email_batches):
        combined_text = "\n".join(batch)
        approx_tokens = len(combined_text.split()) * 4

        n_chunks = max(1, math.ceil(approx_tokens / CHUNK_SIZE))
        chunk_size = max(1, len(batch) // n_chunks)

        chunk_labels = []
        chunk_summaries = []

        for i in range(n_chunks):
            chunk_texts = batch[i * chunk_size: (i + 1) * chunk_size]

            logger.debug(
                "Label generation | batch=%d/%d chunk=%d/%d emails=%d",
                batch_idx + 1,
                len(email_batches),
                i + 1,
                n_chunks,
                len(chunk_texts)
            )

            label, summary = label_prompting(
                emails=chunk_texts,
                role="email clustering"
            )

            chunk_labels.append(label)
            chunk_summaries.append(summary)

        # Merge chunks
        if len(chunk_labels) > 1:
            label, summary = label_prompting(
                labels=chunk_labels,
                summaries=chunk_summaries,
                role="hierarchical email clustering"
            )
        else:
            label, summary = chunk_labels[0], chunk_summaries[0]

        batch_labels.append(label)
        batch_summaries.append(summary)

    # Final merge over batches
    if len(batch_labels) > 1:
        return label_prompting(
            labels=batch_labels,
            summaries=batch_summaries,
            role="hierarchical email clustering"
        )

    return batch_labels[0], batch_summaries[0]


# -------------------------
# Parent cluster labeling
# -------------------------

def generate_parent_label_and_summary(
    child_labels,
    child_summaries
):
    """
    Generate label and summary for a PARENT cluster
    using child labels and summaries.
    """

    if not child_labels:
        return "Uncategorized", ""

    return label_prompting(
        labels=child_labels,
        summaries=child_summaries,
        role="hierarchical email clustering"
    )