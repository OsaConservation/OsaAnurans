"""Build labels from intervals or annotator votes and save immutable snapshots."""

import csv
import json
import logging
import shutil
import tempfile
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .records import IntervalLabel, LabelCounts, WindowRecord

logger = logging.getLogger(__name__)


def aggregate_votes(rows: Iterable[Mapping]) -> list[LabelCounts]:
    """Count annotator votes for each window/species pair.

    Rules:
    - Count every present/absent answer; do not deduplicate annotators.
    - Ignore unsure votes, retaining zero/zero for unsure-only pairs.
    - LabelCounts.response determines strict majority; ties remain unknown.

    Returns:
    - Counts sorted by window ID and species ID.
    """
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        window_id = row["window_id"]
        species_id = row["query_species_id"]
        response = row["response"]
        if not window_id or not species_id:
            raise ValueError("Annotator answers need window_id and query_species_id")
        if response not in {"present", "absent", "unsure"}:
            raise ValueError(f"Unknown annotator response: {response!r}")

        pair = counts[(window_id, species_id)]
        if response == "present":
            pair[0] += 1
        elif response == "absent":
            pair[1] += 1

    return [
        LabelCounts(window_id, species_id, present, absent)
        for (window_id, species_id), (present, absent) in sorted(counts.items())
    ]


def map_intervals_to_windows(
    intervals: Iterable[IntervalLabel], windows: list[WindowRecord]
) -> list[LabelCounts]:
    """Convert CSV presence intervals into window/species labels.

    Steps:
    1. Group windows by recording and sort by source start time.
    2. Find windows with positive-duration overlap for each interval.
    3. Return one present=1 label per unique window/species pair.

    Rules:
    - Boundary-only contact is not overlap.
    - Warn about partial coverage or intervals without a matching window.
    - Leave all other pairs unknown; never infer absence from missing intervals.
    """
    by_recording: dict[str, list[WindowRecord]] = defaultdict(list)
    for window in windows:
        by_recording[window.recording_id].append(window)
    starts = {}
    for recording_id, recording_windows in by_recording.items():
        recording_windows.sort(key=lambda window: window.start_sample)
        starts[recording_id] = [window.start_seconds for window in recording_windows]

    present_pairs: set[tuple[str, str]] = set()
    for interval in intervals:
        recording_windows = by_recording.get(interval.recording_id, [])
        # Windows starting at/after the interval end cannot overlap it.
        stop = bisect_left(starts.get(interval.recording_id, []), interval.end)
        matched = False
        for window in recording_windows[:stop]:
            overlap = min(window.end_seconds, interval.end) - max(
                window.start_seconds, interval.start
            )
            if overlap <= 0:
                continue

            matched = True
            present_pairs.add((window.window_id, interval.species_id))
            if interval.start > window.start_seconds or interval.end < window.end_seconds:
                logger.warning(
                    "Partial annotation overlap: recording=%s species=%s interval=[%s, %s] "
                    "window=%s [%s, %s] overlap=%ss",
                    interval.recording_id,
                    interval.species_id,
                    interval.start,
                    interval.end,
                    window.window_id,
                    window.start_seconds,
                    window.end_seconds,
                    overlap,
                )

        if not matched:
            logger.warning(
                "Annotation has no matching window: recording=%s species=%s interval=[%s, %s]",
                interval.recording_id,
                interval.species_id,
                interval.start,
                interval.end,
            )

    return [
        LabelCounts(window_id, species_id, present=1)
        for window_id, species_id in sorted(present_pairs)
    ]


def _write_csv(path: Path, fields: list[str], rows: Iterable[Mapping]) -> None:
    """Write UTF-8 CSV rows with the requested column order and header."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_labels(labels: list[LabelCounts]) -> None:
    """Require unique nonempty window/species pairs and nonnegative integer counts."""
    seen = set()
    for label in labels:
        key = (label.window_id, label.species_id)
        if not label.window_id or not label.species_id or key in seen:
            raise ValueError(f"Invalid or duplicate snapshot label: {key}")
        if any(type(count) is not int or count < 0 for count in (label.present, label.absent)):
            raise ValueError(f"Vote counts must be nonnegative integers: {key}")
        seen.add(key)


def save_snapshot(
    directory: Path,
    labels: list[LabelCounts],
    source: str,
    raw_answers: Iterable[Mapping] | None = None,
    windows: Iterable[WindowRecord] | None = None,
) -> Path:
    """Save an immutable local label snapshot and return its directory.

    Files:
    - consensus.csv: Present/absent counts by window/species.
    - snapshot.json: Format version, source, creation time, and label count.
    - answers.jsonl: Original answer rows, when supplied.
    - windows.csv: Window metadata, when supplied.

    Write rules:
    - Validate labels and refuse to overwrite an existing snapshot.
    - Write into a temporary sibling directory, then rename after success.
    - Remove the temporary directory on failure or interruption.
    """
    _validate_labels(labels)
    directory = directory.resolve()
    if directory.exists():
        raise FileExistsError(f"Snapshot already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        _write_csv(
            temporary / "consensus.csv",
            ["window_id", "species_id", "present", "absent"],
            (
                asdict(label)
                for label in sorted(labels, key=lambda label: (label.window_id, label.species_id))
            ),
        )
        metadata = {
            "version": 1,
            "source": source,
            "created_at": datetime.now(UTC).isoformat(),
            "label_count": len(labels),
        }
        if raw_answers is not None:
            # JSON preserves all original fields, including provenance added by the cloud adapter.
            with (temporary / "answers.jsonl").open("w", encoding="utf-8") as handle:
                for row in raw_answers:
                    handle.write(json.dumps(dict(row), default=str) + "\n")
        if windows is not None:
            _write_csv(
                temporary / "windows.csv",
                list(WindowRecord.__dataclass_fields__),
                (asdict(window) for window in windows),
            )
        (temporary / "snapshot.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(directory)
    except BaseException:
        shutil.rmtree(temporary)
        raise

    return directory


def load_reference_seed_pairs(directory: Path) -> set[tuple[str, str]]:
    """Read reference/template provenance from the snapshot's original answers.

    - Exclude these window/species pairs from review, regardless of later votes.
    - Ordinary annotations, including Grillos ground truth, remain eligible for uncertainty.
    - CSV snapshots without original answers contain no reference seed provenance.
    """
    path = directory / "answers.jsonl"
    if not path.exists():
        return set()
    pairs = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            answer = json.loads(line)
            if answer.get("annotator_id") in {"new-species", "eoin-template-seed"}:
                pairs.add((answer["window_id"], answer["query_species_id"]))
    return pairs


def load_snapshot(directory: Path) -> list[LabelCounts]:
    """Load vote counts from a saved snapshot.

    Checks:
    - Supported format version and expected number of labels.
    - Unique window/species pairs with nonnegative integer counts.

    Returns:
    - LabelCounts records, retaining ties and unsure-only pairs as unknown.
    """
    metadata = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    if metadata.get("version") != 1:
        raise ValueError("Unsupported annotation snapshot version")
    with (directory / "consensus.csv").open(encoding="utf-8", newline="") as handle:
        labels = [
            LabelCounts(
                row["window_id"],
                row["species_id"],
                int(row["present"]),
                int(row["absent"]),
            )
            for row in csv.DictReader(handle)
        ]
    _validate_labels(labels)
    if len(labels) != metadata.get("label_count"):
        raise ValueError("Annotation snapshot label count does not match its metadata")

    return labels
