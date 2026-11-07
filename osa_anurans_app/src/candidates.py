"""Score all windows and save the lowest-scoring candidates per species."""

import json
from bisect import insort
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

from .model.dataloader import EmbeddingStore
from .model.model import ProbeArtifact
from .records import EmbeddingRow
from .scoring import score_candidate

CANDIDATE_NAMESPACE = UUID("dc93d2af-d257-536a-90b8-30fbf6cbe1dd")


def write_candidates(
    path: Path,
    rows: list[EmbeddingRow],
    artifact: ProbeArtifact,
    batch_size: int,
    candidates_per_species: int = 100,
) -> int:
    """Score all windows and save up to N lowest-scoring candidates per species.

    Steps:
    1. Predict in batches across the full local window pool.
    2. Keep at most N candidates per species; break score ties by window ID.
    3. Write the selection to JSONL, then rename the completed file.

    Notes:
    - Keep all available candidates when a species has fewer than N windows.
    - Retain at most N rows per species in memory, not the entire candidate pool.
    """
    if type(candidates_per_species) is not int or candidates_per_species < 1:
        raise ValueError("candidates_per_species must be a positive integer")
    selected = defaultdict(list)
    for candidate in iter_candidates(rows, artifact, batch_size=batch_size):
        bucket = selected[candidate["query_species_id"]]
        if len(bucket) == candidates_per_species and _candidate_rank(candidate) >= _candidate_rank(
            bucket[-1]
        ):
            continue
        insort(bucket, candidate, key=_candidate_rank)
        if len(bucket) > candidates_per_species:
            bucket.pop()

    temporary = path.with_suffix(".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for species in sorted(selected):
            for candidate in selected[species]:
                handle.write(json.dumps(candidate, allow_nan=False) + "\n")
                count += 1
    temporary.replace(path)
    return count


def _candidate_rank(candidate: dict) -> tuple[float, str]:
    """Sort by uncertainty first, then stable window ID for equal scores."""
    return candidate["score"], candidate["window_id"]


def iter_candidates(
    rows: list[EmbeddingRow],
    artifact: ProbeArtifact,
    batch_size: int = 256,
    created_at: str | None = None,
) -> Iterator[dict]:
    """Yield scores for every local window and trained species before selection.

    Steps:
    1. Require unique window IDs and the classifier's embedding configuration.
    2. Load vectors and predict class probabilities in batches.
    3. Yield window/species IDs, run-specific candidate ID, timestamp, and score.

    Rules:
    - Include unlabeled and already annotated windows.
    - Use one creation timestamp for the whole iteration.
    - write_candidates selects the lowest N per species from these rows.
    """
    if batch_size < 1:
        raise ValueError("Candidate batch size must be positive")
    if not rows:
        raise ValueError("No embedding windows available for candidate generation")
    if any(row.config_id != artifact.config_id for row in rows):
        raise ValueError("Candidate embeddings do not match the probe configuration")
    if len({row.window_id for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate embedding window IDs")

    timestamp = created_at or datetime.now(UTC).isoformat()
    store = EmbeddingStore()
    # Score all windows, including previously reviewed and unlabeled windows.
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        probabilities = artifact.predict_probabilities(store.read_batch(batch))
        for row, values in zip(batch, probabilities, strict=True):
            for species, probability in zip(artifact.species_ids, values, strict=True):
                candidate_id = str(
                    uuid5(
                        CANDIDATE_NAMESPACE,
                        f"{artifact.run_id}:{row.window_id}:{species}",
                    )
                )
                yield {
                    "candidate_id": candidate_id,
                    "window_id": row.window_id,
                    "query_species_id": species,
                    "created_at": timestamp,
                    "score": score_candidate(
                        probability, {"window_id": row.window_id, "species_id": species}
                    ),
                }
