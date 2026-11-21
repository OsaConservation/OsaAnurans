"""Build per-species queues from similar calls and classifier uncertainty."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from heapq import heappush, heapreplace
from pathlib import Path
from uuid import UUID, uuid5

from .files import write_json
from .model.dataloader import EmbeddingStore
from .model.model import ProbeArtifact
from .records import EmbeddingRow, LabelCounts
from .scoring import score_candidate, similarity_fraction
from .similarity import rank_similar_windows

logger = logging.getLogger(__name__)
CANDIDATE_NAMESPACE = UUID("dc93d2af-d257-536a-90b8-30fbf6cbe1dd")


def select_pools(similarity, uncertainty, limit, fraction):
    """Take similarity first, then uncertain unique windows; refill shortages."""
    similarity_count = round(limit * fraction)
    chosen = [(window, "similarity", score) for window, score in similarity[:similarity_count]]
    seen = {window for window, _, _ in chosen}
    uncertain = [(window, score) for window, score in uncertainty if window not in seen]
    chosen.extend(
        (window, "uncertainty", score) for window, score in uncertain[: limit - len(chosen)]
    )
    seen = {window for window, _, _ in chosen}
    for window, score in similarity[similarity_count:]:
        if len(chosen) >= limit:
            break
        if window not in seen:
            chosen.append((window, "similarity", score))
            seen.add(window)
    # Refill may add similarity rows after uncertainty; keep the agreed queue order.
    return [item for item in chosen if item[1] == "similarity"] + [
        item for item in chosen if item[1] == "uncertainty"
    ]


def _positive_seeds(rows, labels):
    """Group present consensus by species; multiple windows count as one recording."""
    by_window = {row.window_id: row for row in rows}
    seeds = defaultdict(dict)
    for label in labels:
        if label.response == "present" and label.window_id in by_window:
            seeds[label.species_id][label.window_id] = by_window[label.window_id]
    return seeds


@dataclass(frozen=True)
class _UncertaintyMatch:
    """Reverse heap ordering so the worst retained score/ID is at the root."""

    window_id: str
    score: float

    def __lt__(self, other):
        return (self.score, self.window_id) > (other.score, other.window_id)


def _uncertainty_pools(rows, artifact, batch_size, candidate_limit, *, excluded_windows=None):
    """Retain at most 2N scores per species, then return them best first.

    - Extra entries allow overlap with up to N similarity selections.
    - Still score every window; discard worse entries immediately.
    - Filter excluded pairs before retaining scores, so they cannot crowd out fresh windows.
    """
    pools = defaultdict(list)
    if artifact is None or not rows:
        return pools
    heaps = defaultdict(list)
    excluded_windows = excluded_windows or {}
    capacity = 2 * candidate_limit
    for window_id, species, score in iter_uncertainty_scores(rows, artifact, batch_size):
        if window_id in excluded_windows.get(species, ()):
            continue
        match = _UncertaintyMatch(window_id, score)
        heap = heaps[species]
        if len(heap) < capacity:
            heappush(heap, match)
        elif heap[0] < match:
            heapreplace(heap, match)
    for species, heap in heaps.items():
        pools[species] = sorted(
            [(match.window_id, match.score) for match in heap], key=lambda item: (item[1], item[0])
        )
    return pools


def _report_candidate_counts(summary: dict) -> None:
    """Show selected counts once, after the candidate files have been saved."""
    print(flush=True)
    logger.info("Candidate summary")
    logger.info("%-10s %12s %12s", "Species", "Similarity", "Uncertainty")
    for species, counts in summary.items():
        logger.info(
            "%-10s %12d %12d",
            species,
            counts["selected_similarity"],
            counts["selected_uncertainty"],
        )
    logger.info(
        "%-10s %12d %12d",
        "Total",
        sum(counts["selected_similarity"] for counts in summary.values()),
        sum(counts["selected_uncertainty"] for counts in summary.values()),
    )


def write_candidates(
    path: Path,
    rows: list[EmbeddingRow],
    artifact: ProbeArtifact | None,
    batch_size: int,
    candidates_per_species: int = 100,
    *,
    labels: list[LabelCounts],
    reference_seed_pairs: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
    held_out_recordings: set[str] | frozenset[str] = frozenset(),
    similarity_balance: float = 25,
    run_id: str | None = None,
) -> int:
    """Build similarity-first queues; save cloud rows and local selection evidence.

    - Count positive recordings once, using consensus for the selected snapshot.
    - Exclude held-out recordings from both seeds and candidate pools.
    - Similarity always excludes resolved window/species pairs.
    - Both methods exclude supplied reference/template seed pairs from review.
      Keep their embeddings as training examples and similarity seeds.
    - Uncertainty may revisit ordinary annotated pairs, regardless of label count.
    - Keep diagnostic fields separate so publication retains its existing schema.
    """
    if type(candidates_per_species) is not int or candidates_per_species < 1:
        raise ValueError("candidates_per_species must be a positive integer")
    similarity_fraction(0, similarity_balance)
    if batch_size < 1 or not rows:
        raise ValueError("Candidate generation requires rows and a positive batch size")
    if len({row.config_id for row in rows}) != 1:
        raise ValueError("Candidate embeddings must share one configuration")
    if len({row.window_id for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate embedding window IDs")
    eligible = [row for row in rows if row.recording_id not in held_out_recordings]
    seeds = _positive_seeds(eligible, labels)
    resolved = defaultdict(set)
    for label in labels:
        if label.response is not None:
            resolved[label.species_id].add(label.window_id)
    for species in sorted({label.species_id for label in labels} - set(seeds)):
        logger.warning("Skipping %s: no positive seeds outside held-out recordings", species)
    if not seeds:
        raise ValueError("No positive seeds available for candidate generation")
    logger.info("Generating candidates...")
    reference_windows = defaultdict(set)
    for window_id, species in reference_seed_pairs:
        reference_windows[species].add(window_id)
    uncertain = _uncertainty_pools(
        eligible,
        artifact,
        batch_size,
        candidates_per_species,
        excluded_windows=reference_windows,
    )
    timestamp = datetime.now(UTC).isoformat()
    run_id = run_id or (artifact.run_id if artifact is not None else path.parent.name)
    candidates, diagnostics, summary = [], [], {}
    for species in sorted(seeds):
        positive_rows = list(seeds[species].values())
        recording_count = len({row.recording_id for row in positive_rows})
        fraction = (
            similarity_fraction(recording_count, similarity_balance)
            if artifact is not None and species in artifact.species_ids
            else 1.0
        )
        # References still supply seeds, but never become review candidates for that species.
        excluded = resolved[species] | reference_windows[species]
        similarity_rows = [row for row in eligible if row.window_id not in excluded]
        similar = rank_similar_windows(similarity_rows, positive_rows, batch_size)
        selected = select_pools(similar, uncertain[species], candidates_per_species, fraction)
        summary[species] = dict(
            positive_windows=len(positive_rows),
            positive_recordings=recording_count,
            similarity_fraction=fraction,
            requested_similarity=round(candidates_per_species * fraction),
            requested_uncertainty=candidates_per_species - round(candidates_per_species * fraction),
            selected_similarity=sum(method == "similarity" for _, method, _ in selected),
            selected_uncertainty=sum(method == "uncertainty" for _, method, _ in selected),
        )
        for position, (window, method, value) in enumerate(selected):
            candidate = dict(
                candidate_id=str(uuid5(CANDIDATE_NAMESPACE, f"{run_id}:{window}:{species}")),
                window_id=window,
                query_species_id=species,
                created_at=timestamp,
                score=position / max(len(selected) - 1, 1),
                selection_method=method,
            )
            candidates.append(candidate)
            diagnostics.append(dict(**candidate, **{method: value}))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, allow_nan=False) + "\n")
    temporary.replace(path)
    write_json(
        path.parent / "candidate-selection.json", dict(species=summary, candidates=diagnostics)
    )
    _report_candidate_counts(summary)
    return len(candidates)


def iter_uncertainty_scores(rows, artifact, batch_size=256):
    """Yield lightweight scores; create candidate IDs only after final selection."""
    if batch_size < 1:
        raise ValueError("Candidate batch size must be positive")
    if not rows:
        raise ValueError("No embedding windows available for candidate generation")
    if any(row.config_id != artifact.config_id for row in rows):
        raise ValueError("Candidate embeddings do not match the probe configuration")
    if len({row.window_id for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate embedding window IDs")

    store = EmbeddingStore()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        probabilities = artifact.predict_probabilities(store.read_batch(batch))
        for row, values in zip(batch, probabilities, strict=True):
            for species, probability in zip(artifact.species_ids, values, strict=True):
                yield row.window_id, species, score_candidate(probability)
