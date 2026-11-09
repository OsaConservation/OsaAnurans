"""Discover recordings and resolve interval labels to their stable identities."""

import csv
import math
from pathlib import Path

from .records import IntervalLabel, RecordingRecord

AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3"})


def annotation_audio_paths(
    filename: str, csv_directory: Path, roots: tuple[Path, ...]
) -> set[Path]:
    """List possible local paths for a CSV audiofilename.

    Rules:
    - Absolute filename: return its resolved path.
    - Relative filename: try the CSV directory and every configured audio root.
    - Return candidates only; existence and ambiguity are checked by callers.
    """
    relative = Path(filename.strip().replace("\\", "/"))
    if relative.is_absolute():
        return {relative.resolve()}
    return {(root / relative).resolve() for root in (csv_directory, *roots)}


def discover_recordings(roots: tuple[Path, ...]) -> list[Path]:
    """Find audio files under configured directories.

    Steps:
    1. Require every root to be a directory.
    2. Search recursively for supported audio suffixes, ignoring case.
    3. Deduplicate resolved paths and return a stable sorted list.
    """
    paths: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Audio root is not a directory: {root}")

        paths.update(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )

    return sorted(paths, key=lambda path: str(path).casefold())


def _resolve_recording(
    filename: str,
    csv_directory: Path,
    roots: tuple[Path, ...],
    by_path: dict[Path, RecordingRecord],
) -> RecordingRecord:
    """Match a CSV filename to one registered recording.

    Rules:
    - Multiple paths to identical recording contents count as one identity.
    - Missing matches or matches to different identities raise an error.
    """
    candidates = annotation_audio_paths(filename, csv_directory, roots)

    matches = {by_path[path].recording_id: by_path[path] for path in candidates if path in by_path}
    if not matches:
        raise ValueError(f"Annotation audio is not a registered recording: {filename}")
    if len(matches) > 1:
        raise ValueError(f"Annotation audio matches multiple recording roots: {filename}")

    return next(iter(matches.values()))


def read_interval_labels(
    csv_path: Path,
    recordings: list[RecordingRecord],
    roots: tuple[Path, ...],
    species_ids: set[str] | None = None,
) -> list[IntervalLabel]:
    """Read CSV presence intervals and replace filenames with recording IDs.

    Steps:
    1. Require audiofilename, start, end, and label:species columns.
    2. Match each audio path to a registered recording.
    3. Validate species and finite bounds within the source duration.

    Returns:
    - IntervalLabel records in source seconds; no absent labels are inferred.

    Unhappy paths:
    - Invalid rows report the CSV path and line number.
    """
    by_path = {recording.path.resolve(): recording for recording in recordings}
    labels = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audiofilename", "start", "end", "label:species"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Annotation CSV requires columns: {', '.join(sorted(required))}")

        for line_number, row in enumerate(reader, start=2):
            try:
                recording = _resolve_recording(
                    row["audiofilename"].strip(), csv_path.parent, roots, by_path
                )
                start = float(row["start"])
                end = float(row["end"])
                species = row["label:species"].strip()
                if not species or (species_ids is not None and species not in species_ids):
                    raise ValueError(f"Unknown species ID: {species!r}")
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError(f"Invalid interval bounds: {start}, {end}")
                if end > recording.frames / recording.sample_rate + 1e-6:
                    raise ValueError(f"Interval ends after the recording: {end}")
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError(f"{csv_path}:{line_number}: {error}") from error

            labels.append(IntervalLabel(recording.recording_id, species, start, end))

    return labels
