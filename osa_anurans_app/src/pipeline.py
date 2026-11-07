"""Read recordings/labels, generate local training data, and train classifiers.

Workflows:
- import_local: Audio + CSV -> local WAVs, embeddings, and label snapshot.
- train_local: Embeddings + snapshot -> classifier, metrics, and candidates.
- refresh_annotations: BigQuery answers -> local consensus snapshot.

Responsibilities:
- Run stages in order and save counts, errors, and selected input paths.
- Delegate audio, labels, training, and storage to their modules.
- Let publish.py upload WAVs to GCS and write window/candidate rows to BigQuery.

Outputs:
- Each import/training run has its own directory.
- latest.json points to the default snapshot and embedding configuration.

Boundaries:
- Only refresh_annotations reads from GCP here.
- Source audio is never edited.
"""

import csv
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from tqdm import tqdm

from .annotations import (
    aggregate_votes,
    load_reference_seed_pairs,
    load_snapshot,
    map_intervals_to_windows,
    save_snapshot,
)
from .audio import identify_recording, recording_windows, valid_window_audio, write_window_audio
from .catalog import Catalog
from .config import AppConfig, validate_import_sources
from .files import write_json
from .records import IntervalLabel, RecordingRecord
from .sources import annotation_audio_paths, discover_recordings, read_interval_labels

logger = logging.getLogger(__name__)


def new_run_id() -> str:
    """Create a readable UTC run name with a random suffix to avoid collisions."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def _limited_paths(paths: list[Path], config: AppConfig, limit: int | None) -> list[Path]:
    """Select a small import sample, preferring annotated recordings by species.

    Steps:
    1. With no limit, return all discovered paths.
    2. Otherwise, take turns selecting recordings across species.
    3. Remove duplicate paths and fill remaining slots with other recordings.

    Notes:
    - Intended for smoke tests, not representative sampling.
    - A supplied limit must be positive.
    """
    if limit is None:
        return paths
    if limit < 1:
        raise ValueError("Recording limit must be positive")
    if config.annotations_csv is None:
        return paths[:limit]
    by_species: dict[str, list[Path]] = {}
    available = set(paths)
    with config.annotations_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            matches = (
                annotation_audio_paths(
                    row["audiofilename"], config.annotations_csv.parent, config.audio_roots
                )
                & available
            )
            if len(matches) == 1:
                bucket = by_species.setdefault(row["label:species"], [])
                path = matches.pop()
                if path not in bucket:
                    bucket.append(path)
    selected = []
    groups = list(by_species.values())
    # Round-robin selection avoids spending the entire limit on the first species.
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group) and group[index] not in selected:
                selected.append(group[index])
    selected.extend(path for path in paths if path not in selected)
    return selected[:limit]


def _cached_artifact(
    catalog: Catalog, recording, config_id: str, expected: int, dimension: int | None = None
) -> bool:
    """Check whether this recording's embeddings and window WAVs can be reused.

    Checks:
    - Expected number of indexed rows.
    - Finite floating-point matrix with the expected shape.
    - Valid WAV headers for every indexed window.

    Returns:
    - True if reusable; False if missing/corrupt so the caller can rebuild.
    """
    rows = [
        row
        for row in catalog.embedding_rows(config_id)
        if row.recording_id == recording.recording_id
    ]
    if len(rows) != expected:
        return False
    try:
        array = np.load(rows[0].path, mmap_mode="r", allow_pickle=False)
        return (
            array.ndim == 2
            and len(array) == expected
            and array.shape[1] > 0
            and (dimension is None or array.shape[1] == dimension)
            and np.issubdtype(array.dtype, np.floating)
            and np.isfinite(array).all()
            and all(valid_window_audio(catalog.audio_path(row.window_id)) for row in rows)
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def _store_recording(catalog: Catalog, recording, backend, artifacts: Path, *, frames=None) -> bool:
    """Generate and register one recording's aligned audio/vector artifacts.

    Steps:
    1. Reuse saved WAVs and embeddings if they pass validation.
    2. Prepare frames and embed them in window order.
    3. Save those exact frames as WAVs and vectors as one NumPy file.
    4. Store window-to-WAV and window-to-vector-row mappings in SQLite.

    Returns:
    - True if generated; False if reused.

    Options:
    - frames: Reuse already validated model input (e.g. a new species reference).
    """
    windows = recording_windows(recording)
    if _cached_artifact(
        catalog,
        recording,
        backend.config_id,
        len(windows),
        backend.metadata.get("embedding_dimension"),
    ):
        return False
    if frames is None:
        frames = backend.prepare_audio(recording.path)
    if len(frames) != len(windows):
        raise ValueError(
            f"Audio/window alignment mismatch for {recording.path}: {len(frames)} vs {len(windows)}"
        )
    embeddings = backend.embed_frames(frames)
    if embeddings.ndim != 2 or len(embeddings) != len(windows) or not np.isfinite(embeddings).all():
        raise ValueError(f"Embedding/window alignment mismatch for {recording.path}")
    # Save the exact frames used for inference, including wrap padding.
    audio_paths = []
    for window, frame in zip(windows, frames, strict=True):
        audio_path = artifacts / "windows" / "v1" / f"{window.window_id}.wav"
        write_window_audio(audio_path, frame)
        audio_paths.append(audio_path)
    path = (
        artifacts / "embeddings" / "perch_v2" / backend.config_id / f"{recording.recording_id}.npy"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, embeddings.astype(np.float32, copy=False), allow_pickle=False)
    temporary.replace(path)
    # Commit row mappings only after every audio file and the array are complete.
    catalog.register_artifact(windows, backend.config_id, path, audio_paths)
    return True


def _preflight_annotations(
    config: AppConfig,
    recordings: list[RecordingRecord],
    directory: Path,
    allow_missing: bool,
) -> list[IntervalLabel]:
    """Validate CSV labels against selected readable recordings before inference.

    Steps:
    1. Select rows referencing available recordings.
    2. Save selected rows and exclusions for review.
    3. Resolve paths and validate interval bounds.

    Returns:
    - Presence intervals; window labels are created after embedding.

    Unhappy paths:
    - Unavailable rows fail unless allow_missing is set.
    - Limited runs and --skip-invalid allow these exclusions.
    - Invalid selected intervals still fail.
    """
    available = {recording.path.resolve() for recording in recordings}
    kept, excluded = [], []
    with config.annotations_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not {"audiofilename", "start", "end", "label:species"}.issubset(fields or []):
            raise ValueError("Annotation CSV requires audiofilename,start,end,label:species")
        for row in reader:
            candidates = annotation_audio_paths(
                row["audiofilename"], config.annotations_csv.parent, config.audio_roots
            )
            if candidates & available:
                kept.append(row)
            else:
                excluded.append(row)
    directory.mkdir(parents=True, exist_ok=False)
    csv_path = directory / "input.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    excluded_path = directory / "excluded_annotations.json"
    write_json(excluded_path, excluded)
    if excluded and not allow_missing:
        raise ValueError(
            f"{len(excluded)} annotations reference unavailable audio; "
            f"see {excluded_path}. Use --skip-invalid to exclude them explicitly."
        )
    if excluded:
        logger.warning(
            "Excluded %d annotation rows outside available/selected recordings", len(excluded)
        )
    # The copied CSV retains paths relative to its original location.
    roots = (config.annotations_csv.parent, *config.audio_roots)
    return read_interval_labels(csv_path, recordings, roots)


def _bootstrap_labels(config, catalog, snapshot, intervals, completed_window_ids):
    """Map initial presence intervals onto successfully processed windows.

    Steps:
    1. Select windows from recordings that succeeded in this run (new or cached).
    2. Report intervals belonging to failed recordings.
    3. Map presence intervals to windows and save a snapshot.

    Rules:
    - Exclude catalog windows outside this run's selected recordings.
    - Leave unlabeled regions unknown, not absent.

    Returns:
    - Path to the saved label snapshot directory.
    """
    windows = [window for window in catalog.windows() if window.window_id in completed_window_ids]
    completed_recordings = {window.recording_id for window in windows}
    kept = [label for label in intervals if label.recording_id in completed_recordings]
    failed = [label for label in intervals if label.recording_id not in completed_recordings]
    write_json(snapshot / "failed_recording_annotations.json", [asdict(label) for label in failed])
    if failed:
        logger.warning(
            "Excluded %d interval labels whose recordings failed processing", len(failed)
        )
    labels = map_intervals_to_windows(kept, windows)
    return save_snapshot(snapshot / "labels", labels, str(config.annotations_csv), windows=windows)


def _index_recordings(paths: list[Path], catalog: Catalog, report: dict) -> list[RecordingRecord]:
    """Read source metadata, derive content identities, and index valid files.

    Steps:
    1. Read sample rates/frame counts and hash each source file.
    2. Store readable recordings' IDs, paths, hashes, and audio metadata in SQLite.
    3. Collect failures in the shared report.

    Returns:
    - Readable recording records.

    Notes:
    - import_local decides whether indexing failures stop the run.
    - No window decoding or Perch inference occurs here.
    """
    recordings = []
    logger.info("Indexing %d recordings", len(paths))
    for path in tqdm(paths, desc="Indexing recordings", disable=not sys.stderr.isatty()):
        try:
            recording = identify_recording(path)
            catalog.add_recording(recording)
            recordings.append(recording)
        except Exception as error:
            report["failures"].append({"path": str(path), "error": str(error)})
            logger.warning("Cannot index %s: %s", path, error)
    return recordings


def _embed_recordings(
    recordings: list[RecordingRecord],
    catalog: Catalog,
    backend,
    artifacts: Path,
    report: dict,
    report_path: Path,
    skip_invalid: bool,
) -> set[str]:
    """Generate/reuse window WAVs and embeddings for each selected recording.

    Steps:
    1. Process one recording at a time.
    2. Update generated/reused counts in the shared report.

    Returns:
    - IDs of recordings that succeeded, including cache hits.

    Unhappy paths:
    - Save failure details before raising.
    - Continue past recording failures only when skip_invalid is set.
    - Keep completed artifacts for reuse after interruption.
    """
    completed = set()
    logger.info("Generating or reusing embeddings for %d recordings", len(recordings))
    for recording in tqdm(
        recordings, desc="Generating Perch embeddings", disable=not sys.stderr.isatty()
    ):
        try:
            generated = _store_recording(catalog, recording, backend, artifacts)
            report["generated" if generated else "reused"] += 1
            completed.add(recording.recording_id)
        except Exception as error:
            report["failures"].append({"path": str(recording.path), "error": str(error)})
            write_json(report_path, report)
            if not skip_invalid:
                raise
            logger.warning("Cannot process %s: %s", recording.path, error)
    return completed


def import_local(
    config: AppConfig,
    *,
    backend=None,
    limit_recordings: int | None = None,
    skip_invalid: bool = False,
) -> Path | None:
    """Import source audio and CSV labels locally. No GCP access.

    Steps:
    1. Validate inputs and store recording IDs/paths/audio metadata in SQLite.
    2. Generate or reuse window WAVs and embeddings.
    3. With an optional CSV, save presence labels for successful recordings.
    4. Update latest.json with the snapshot and embedding configuration.

    Returns:
    - New CSV snapshot, or the existing snapshot for audio-only imports.
    - None if no labels have been imported or fetched yet.

    Unhappy paths:
    - Invalid inputs stop the run by default.
    - skip_invalid allows reported exclusions, not all errors.
    - Failed imports keep saved WAVs and embeddings for reuse.

    Options:
    - backend: BacpipePerch by default; tests can supply a tiny substitute.
    - limit_recordings: Process a small sample for a smoke test.
    - skip_invalid: Skip unreadable recordings or unavailable CSV audio paths.
    """
    # 1. Discover inputs and create an import-specific progress report.
    validate_import_sources(config)
    run_id = new_run_id()
    report_dir = config.artifacts / "imports" / run_id
    report_dir.mkdir(parents=True)
    paths = _limited_paths(discover_recordings(config.audio_roots), config, limit_recordings)
    report = {
        "run_id": run_id,
        "selected_recordings": len(paths),
        "failures": [],
        "generated": 0,
        "reused": 0,
        "local_only": True,
    }
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        # 2. Hash/index source files and validate CSV paths/intervals before inference.
        recordings = _index_recordings(paths, catalog, report)
        write_json(report_dir / "import.json", report)
        if report["failures"] and not skip_invalid:
            raise ValueError(
                f"Invalid recordings; see {report_dir / 'import.json'}. Use --skip-invalid to continue explicitly."
            )
        if not recordings:
            raise ValueError("No readable recordings selected")
        # Reject bad input before downloading or initializing the Perch model.
        try:
            intervals = (
                _preflight_annotations(
                    config,
                    recordings,
                    report_dir / "annotations",
                    skip_invalid or limit_recordings is not None,
                )
                if config.annotations_csv is not None
                else []
            )
        except (ValueError, OSError) as error:
            report.update(state="annotation_preflight_failed", annotation_error=str(error))
            write_json(report_dir / "import.json", report)
            raise
        # 3. Initialize Perch, then generate/reuse window WAVs and embedding arrays.
        if backend is None:
            from .bacpipe_adapter import BacpipePerch

            backend = BacpipePerch(config.artifacts, config.device)
        write_json(
            config.artifacts / "embeddings" / "perch_v2" / backend.config_id / "config.json",
            backend.metadata,
        )
        completed = _embed_recordings(
            recordings,
            catalog,
            backend,
            config.artifacts,
            report,
            report_dir / "import.json",
            skip_invalid,
        )
        # 4. Label windows whose recordings succeeded in this run (new or cached).
        # Exclude failed recordings and catalog entries outside this run's inputs,
        # e.g. recordings omitted by --limit-recordings or changed audio roots.
        rows = catalog.embedding_rows(backend.config_id)
        completed_window_ids = {row.window_id for row in rows if row.recording_id in completed}
        if not completed_window_ids:
            report["state"] = "no_completed_recordings"
            write_json(report_dir / "import.json", report)
            raise ValueError("No recordings completed embedding generation; see import report")
        latest_path = config.artifacts / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else {}
        snapshot = Path(latest["snapshot"]) if latest.get("snapshot") else None
        if config.annotations_csv is not None:
            snapshot = _bootstrap_labels(
                config, catalog, report_dir / "annotations", intervals, completed_window_ids
            )
            latest["snapshot"] = str(snapshot)
        else:
            logger.info(
                "Audio-only import: keeping the saved label snapshot; new windows are unknown"
            )
        # 5. Record success and make this snapshot/configuration the training defaults.
        report.update(
            {
                "config_id": backend.config_id,
                "snapshot": str(snapshot) if snapshot else None,
                "catalog_windows": len(rows),
                "state": "complete",
            }
        )
        write_json(report_dir / "import.json", report)
        latest["config_id"] = backend.config_id
        write_json(latest_path, latest)
    logger.info("Local import complete: %s", report_dir)
    return snapshot


def refresh_annotations(config: AppConfig, bq) -> Path:
    """Read annotator answers from BigQuery and return a new local snapshot.

    Steps:
    1. Fetch answers using the same window IDs as the local catalog.
    2. Count present/absent votes; ignore unsure.
    3. Save vote counts, original answers, and IDs missing from the local catalog.
    4. Update latest.json to select the new snapshot.

    Returns:
    - Path to the saved label snapshot directory.

    Boundaries:
    - No CSV label merging, model retraining, or cloud writes.
    - Unmatched answers are retained but need local embeddings to train.
    """
    logger.info("Fetching annotator answers from BigQuery...")
    answers = bq.fetch_annotations()
    logger.info("Fetched %d answers; counting votes and saving a local snapshot...", len(answers))
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        local_window_ids = {window.window_id for window in catalog.windows()}
        unmatched = sorted(
            {row["window_id"] for row in answers if row["window_id"] not in local_window_ids}
        )
        if unmatched:
            logger.warning(
                "BigQuery annotation refresh contains %d window IDs absent from the local catalog; "
                "all vote types are retained in the snapshot and unmatched report",
                len(unmatched),
            )
        # Keep unmatched answers too; training later requires a matching embedding.
        labels = aggregate_votes(answers)
        snapshot = save_snapshot(
            config.artifacts / "annotations" / new_run_id(),
            labels,
            "bigquery",
            raw_answers=answers,
            windows=catalog.windows(),
        )
        write_json(
            snapshot / "unmatched_windows.json",
            {
                "window_ids": unmatched,
                "answer_count": sum(row["window_id"] not in local_window_ids for row in answers),
            },
        )
    # Change the label default while retaining the selected embedding configuration.
    latest_path = config.artifacts / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else {}
    latest["snapshot"] = str(snapshot)
    write_json(latest_path, latest)
    logger.info("Annotation refresh complete: %d window/species pairs; %s", len(labels), snapshot)
    return snapshot


def train_local(
    config: AppConfig, *, snapshot: Path | None = None, config_id: str | None = None
) -> Path:
    """Train from saved local data and return the new model/candidate run directory.

    Steps:
    1. Select saved embeddings and a label snapshot.
    2. Select windows with one known-present species; keep recordings in one split.
    3. Train/evaluate the classifier; save weights, species order, and metrics.
    4. Select similar/uncertain windows; save up to N candidates per species.

    Returns:
    - Path to the new model/candidate run directory.

    Options:
    - snapshot/config_id: Override latest.json defaults.

    Boundaries:
    - No original audio required and no cloud writes.
    """
    from .candidates import write_candidates
    from .model.datasets import DatasetSplit, build_dataset, split_by_recording
    from .model.train import train_classifier

    # 1. Select the saved inputs; original source audio is not needed for training.
    latest_path = config.artifacts / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else {}
    snapshot = snapshot or (Path(latest["snapshot"]) if "snapshot" in latest else None)
    config_id = config_id or latest.get("config_id")
    if snapshot is None or config_id is None:
        raise ValueError(
            "Import audio and refresh labels from BigQuery first, "
            "or specify --snapshot and --embedding-config"
        )
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        rows = catalog.embedding_rows(config_id)

    # 2. Select single-label examples, train the probe, and save weights/metrics.
    labels = load_snapshot(snapshot)
    dataset = build_dataset(rows, labels)
    run_dir = config.artifacts / "runs" / new_run_id()
    # Reuse exactly the same split for model evaluation and candidate exclusion.
    if len({row.recording_id for row in dataset.rows}) >= 2:
        split = split_by_recording(dataset, config.test_fraction, config.seed)
    else:
        split = DatasetSplit(list(range(len(dataset.rows))), [])
    held_out = {dataset.rows[index].recording_id for index in split.test}
    artifact = None
    if len({dataset.labels[index] for index in split.train}) >= 2:
        artifact = train_classifier(dataset, run_dir, config, split=split)
    else:
        logger.warning("Not enough classes for a probe; generating similarity-only candidates")
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", {"mode": "similarity_only", "config_id": config_id})
    write_json(run_dir / "annotations.json", {"snapshot": str(snapshot.resolve())})

    # 3. Select similarity matches first, then uncertainty; never query the cloud.
    count = write_candidates(
        run_dir / "candidates.jsonl",
        rows,
        artifact,
        config.batch_size,
        config.candidates_per_species,
        labels=labels,
        reference_seed_pairs=load_reference_seed_pairs(snapshot),
        held_out_recordings=held_out,
        similarity_balance=config.similarity_balance,
        run_id=run_dir.name,
    )
    write_json(
        run_dir / "candidates.json",
        {
            "count": count,
            "window_count": len(rows),
            "candidates_per_species": config.candidates_per_species,
            "species_ids": sorted(
                json.loads((run_dir / "candidate-selection.json").read_text())["species"]
            ),
            "classifier_available": artifact is not None,
            "local_only": True,
        },
    )
    logger.info("Local run with %d candidates saved: %s", count, run_dir)
    return run_dir
