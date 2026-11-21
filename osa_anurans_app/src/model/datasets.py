"""Build single-label training examples and keep recordings on one split side."""

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

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


def _test_recordings(dataset: TrainingDataset, test_fraction: float, seed: int) -> set[str]:
    """Meet per-species recording quotas without splitting shared recordings.

    - Handle scarce species first; one-recording species have a zero test quota.
    - Prefer recordings that keep each species' window counts near its target share.
    - Use stable hash ordering so an unrelated new species does not reshuffle others.
    """
    by_recording = defaultdict(Counter)
    by_species = defaultdict(set)
    for row, species in zip(dataset.rows, dataset.labels, strict=True):
        by_recording[row.recording_id][species] += 1
        by_species[species].add(row.recording_id)
    targets = {
        species: min(len(recordings) - 1, max(1, round(len(recordings) * test_fraction)))
        for species, recordings in by_species.items()
    }
    ranks = {
        recording: hashlib.sha256(f"{seed}:{recording}".encode()).digest()
        for recording in by_recording
    }
    selected, test_counts, test_windows = set(), Counter(), Counter()
    total_windows = Counter(dataset.labels)
    for species in sorted(by_species, key=lambda item: (len(by_species[item]), item)):
        while test_counts[species] < targets[species]:
            available = [
                recording
                for recording in by_species[species] - selected
                if all(test_counts[item] < targets[item] for item in by_recording[recording])
            ]
            if not available:
                break
            # Aim for the average number of windows per recording at each selection.
            desired = total_windows[species] / len(by_species[species]) * (test_counts[species] + 1)
            recording = min(
                available,
                key=lambda item: (
                    abs(test_windows[species] + by_recording[item][species] - desired),
                    ranks[item],
                ),
            )
            selected.add(recording)
            test_counts.update(by_recording[recording].keys())
            test_windows.update(by_recording[recording])
        if test_counts[species] < targets[species]:
            logger.warning(
                "%s: held out %d of %d target recordings; shared recordings limit the split",
                species,
                test_counts[species],
                targets[species],
            )
    return selected


def split_by_recording(
    dataset: TrainingDataset, test_fraction: float = 0.2, seed: int = 42
) -> DatasetSplit:
    """Split per species, assigning whole recordings and retaining training coverage.

    Rules:
    - Keep adjacent windows from the same recording on one side.
    - Require at least two labeled recordings and a test fraction in (0, 1).
    - Target test_fraction of each species' labeled recordings, with at least one in training.
    - Keep a species' only recording in training; test metrics remain unavailable.
    - Shared recordings can prevent exact per-species quotas; coverage takes priority.
    - If none can be held out, train on all recordings and report no evaluation.
    - Approximate window balance; no separation by deployment/site/date is promised.

    Returns:
    - Row indices for each partition, preserving original example order.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    recordings = sorted({row.recording_id for row in dataset.rows})
    if len(recordings) < 2:
        raise ValueError("Need labeled windows from at least two recordings for a held-out split")

    test_recordings = _test_recordings(dataset, test_fraction, seed)
    if not test_recordings:
        logger.warning(
            "No recording can be held out while keeping every species in training. "
            "Using all labeled recordings for training; held-out evaluation is unavailable. "
            "Add labeled recordings to enable evaluation."
        )
    train = []
    test = []
    for index, row in enumerate(dataset.rows):
        (test if row.recording_id in test_recordings else train).append(index)
    return DatasetSplit(train, test)
