"""Exact cosine search over local vectors; no persistent vector index needed."""

import logging

import numpy as np

from .model.dataloader import EmbeddingStore

logger = logging.getLogger(__name__)


def normalized_vectors(rows, store):
    """Return unit vectors and their rows; skip zero vectors, reject invalid data."""
    vectors = store.read_batch(rows).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    if not valid.all():
        logger.warning("Skipping %d zero-norm similarity vectors", int((~valid).sum()))
    return [row for row, keep in zip(rows, valid, strict=True) if keep], vectors[valid] / norms[
        valid, None
    ]


def rank_similar_windows(rows, seeds, batch_size=256):
    """Rank by maximum seed cosine similarity, excluding the seed windows.

    - Batch both dimensions to bound temporary matrix memory.
    - Return (window ID, similarity), highest first; ties use window ID.
    """
    if batch_size < 1:
        raise ValueError("Similarity batch size must be positive")
    if not rows or not seeds:
        return []
    if len({row.config_id for row in [*rows, *seeds]}) != 1:
        raise ValueError("Similarity embeddings must share one configuration")
    store = EmbeddingStore()
    seed_ids = {row.window_id for row in seeds}
    seed_batches = []
    for start in range(0, len(seeds), batch_size):
        _, vectors = normalized_vectors(seeds[start : start + batch_size], store)
        if len(vectors):
            seed_batches.append(vectors)
    if not seed_batches:
        logger.warning("No nonzero positive seeds available for similarity search")
        return []
    eligible = [row for row in rows if row.window_id not in seed_ids]
    ranked = []
    for start in range(0, len(eligible), batch_size):
        batch, vectors = normalized_vectors(eligible[start : start + batch_size], store)
        if not batch:
            continue
        scores = np.full(len(batch), -np.inf)
        for seed_vectors in seed_batches:
            scores = np.maximum(scores, (vectors @ seed_vectors.T).max(axis=1))
        ranked.extend(
            (row.window_id, float(np.clip(score, -1, 1)))
            for row, score in zip(batch, scores, strict=True)
        )
    return sorted(ranked, key=lambda item: (-item[1], item[0]))
