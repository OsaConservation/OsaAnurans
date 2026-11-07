"""Build single-label training examples and keep recordings on one split side."""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from ..records import EmbeddingRow, LabelCounts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingDataset:
    """Hold selected single-label examples without loading their vectors.

    Fields:
    - rows/labels: Matching embedding locations and target species IDs by index.
    - species_ids: Sorted species represented in the selected labels.
    - config_id: Shared embedding configuration.
    - excluded: Counts of windows omitted by the single-label rules.
    """

    rows: list[EmbeddingRow]
    labels: list[str]
    species_ids: tuple[str, ...]
    config_id: str
    excluded: dict[str, int]


@dataclass(frozen=True)
class DatasetSplit:
    """Store train/test indices into TrainingDataset.rows and its matching labels."""

    train: list[int]
    test: list[int]


def build_dataset(rows: list[EmbeddingRow], labels: list[LabelCounts]) -> TrainingDataset:
    """Select local windows with exactly one known-present species.

    Steps:
    1. Require unique windows/vector locations from one embedding configuration.
    2. Collect species with present majority for each window.
    3. Keep one-positive windows; count unknown/absent-only/multi-positive exclusions.

    Rules:
    - Warn about labels without matching local vectors; ignore them for training.
    - A missing positive label does not establish true absence.
    - Return locations and species IDs; vector reads happen in the dataloader.
    """
    if not rows:
        raise ValueError("No embeddings available for training")
    configs = {row.config_id for row in rows}
    if len(configs) != 1:
        raise ValueError("Select exactly one embedding configuration for training")
    window_ids = [row.window_id for row in rows]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("A window has multiple embedding rows in the selected configuration")
    locations = {(row.path.resolve(), row.row_index) for row in rows}
    if len(locations) != len(rows):
        raise ValueError("Multiple windows map to the same embedding file row")

    positives: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for label in labels:
        key = (label.window_id, label.species_id)
        if key in seen:
            raise ValueError(f"Duplicate consensus label: {key}")
        if label.present < 0 or label.absent < 0:
            raise ValueError(f"Negative annotation vote count: {key}")
        seen.add(key)
        if label.response == "present":
            positives[label.window_id].add(label.species_id)

    missing = set(positives) - set(window_ids)
    if missing:
        logger.warning("Ignoring annotations for %d windows without local embeddings", len(missing))

    selected_rows = []
    selected_labels = []
    excluded = {"unknown_or_absent_only": 0, "multiple_present_species": 0}
    for row in rows:
        species = positives.get(row.window_id, set())
        if not species:
            excluded["unknown_or_absent_only"] += 1
            continue
        if len(species) > 1:
            excluded["multiple_present_species"] += 1
            continue
        selected_rows.append(row)
        selected_labels.append(next(iter(species)))

    logger.info("Single-label examples: %d; exclusions: %s", len(selected_rows), excluded)
    return TrainingDataset(
        selected_rows,
        selected_labels,
        tuple(sorted(set(selected_labels))),
        next(iter(configs)),
        excluded,
    )


def split_by_recording(
    dataset: TrainingDataset, test_fraction: float = 0.2, seed: int = 42
) -> DatasetSplit:
    """Assign whole recordings while keeping every species in training.

    Rules:
    - Keep adjacent windows from the same recording on one side.
    - Require at least two labeled recordings and a test fraction in (0, 1).
    - Move a recording to test only if every species remains in training.
    - The requested test fraction is a target; coverage takes priority.
    - If none can be held out, train on all recordings and report no evaluation.
    - Do not promise class balance or separation by deployment/site/date.

    Returns:
    - Row indices for each partition, preserving original example order.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    recordings = sorted({row.recording_id for row in dataset.rows})
    if len(recordings) < 2:
        raise ValueError("Need labeled windows from at least two recordings for a held-out split")

    shuffled = np.random.default_rng(seed).permutation(recordings)
    test_count = max(1, min(len(recordings) - 1, round(len(recordings) * test_fraction)))
    species_by_recording: dict[str, set[str]] = defaultdict(set)
    for row, species in zip(dataset.rows, dataset.labels, strict=True):
        species_by_recording[row.recording_id].add(species)
    # Count supporting recordings, rather than windows, to avoid leakage.
    remaining = Counter(
        species for species_set in species_by_recording.values() for species in species_set
    )
    test_recordings = set()
    for recording in shuffled:
        species = species_by_recording[recording]
        if any(remaining[item] <= 1 for item in species):
            continue
        test_recordings.add(recording)
        remaining.subtract(species)
        if len(test_recordings) == test_count:
            break
    if not test_recordings:
        logger.warning(
            "No recording can be held out while keeping every species in training. "
            "Using all labeled recordings for training; held-out evaluation is unavailable. "
            "Add labeled recordings to enable evaluation."
        )
    elif len(test_recordings) < test_count:
        logger.warning(
            "Held out %d of %d requested recordings to keep every species in training",
            len(test_recordings),
            test_count,
        )
    train = []
    test = []
    for index, row in enumerate(dataset.rows):
        (test if row.recording_id in test_recordings else train).append(index)
    return DatasetSplit(train, test)
