"""Five-second references -> local vectors, cloud catalog rows, and seed votes.

Rules:
- Validate the whole manifest before cloud writes.
- Reuse stable identities; retries never add another seed vote.
- Refresh the full BigQuery snapshot before allowing training to continue.
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import soundfile as sf

from .audio import file_hash, identify_recording, recording_windows
from .catalog import Catalog
from .cloud.species import insert_batch, read_catalog
from .files import write_json
from .pipeline import _store_recording, new_run_id, refresh_annotations

logger = logging.getLogger(__name__)
REFERENCE_NAMESPACE = UUID("baae4a91-5fd8-4538-bc67-d175d097dc84")
SEED_NAMESPACE = UUID("575b5c46-5370-496d-b0bc-a26e4468ac32")
SUBMISSION_NAMESPACE = UUID("964804bd-ac35-451a-9ca3-04a9135234bf")
AUDIO_MATCH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SpeciesInput:
    """Reviewed metadata and source bytes; source relocation does not change IDs."""

    species_id: str
    scientific_name: str
    common_name: str
    audio: Path
    content_hash: str


@dataclass(frozen=True)
class SpeciesManifest:
    """Immutable review payload; verify hashes again before processing."""

    path: Path
    content_hash: str
    entries: tuple[SpeciesInput, ...]


def normalize_name(value: str) -> str:
    """Require a binomial; canonicalize genus/epithet capitalization."""
    words = value.split()
    if len(words) != 2 or any(len(word) < 3 or not word.isalpha() for word in words):
        raise ValueError("scientific_name must contain two alphabetic words (genus and species)")
    return f"{words[0].capitalize()} {words[1].lower()}"


def validate_audio(path: Path) -> None:
    """Check decoded samples: exactly five seconds, finite, and nonzero energy."""
    if path.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError(f"reference_audio must be WAV or MP3: {path}")
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot decode reference_audio: {path}") from exc
    if len(audio) != 5 * rate:
        raise ValueError(f"reference_audio must be exactly five decoded seconds: {path}")
    if not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError(f"reference_audio must contain finite, nonzero samples: {path}")
    if not np.any(audio.mean(axis=1)):
        raise ValueError(f"reference_audio channels cancel during mono conversion: {path}")


def load_manifest(path: Path) -> SpeciesManifest:
    """Validate JSON, names, paths, duplicate pairs, and every source recording.

    - Resolve relative paths beside the manifest; accept UTF-8 with a BOM.
    - Capture file hashes so a later edit invalidates the reviewed inputs.
    - No model initialization, catalog changes, or cloud writes.
    """
    path = path.expanduser().resolve()
    digest = file_hash(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data) != {"species"}:
        raise ValueError("Manifest must contain exactly a species array")
    rows = data["species"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("species must be a nonempty array")
    entries, names, pairs = [], {}, set()
    fields = {"scientific_name", "common_name", "reference_audio"}
    for index, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError(f"Expected fields: {', '.join(sorted(fields))}")
            for field in fields:
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"{field} must be a nonempty string")
            scientific = normalize_name(row["scientific_name"])
            genus, epithet = scientific.split()
            species_id = (genus[:3] + epithet[:3]).upper()
            common = " ".join(row["common_name"].split())
            metadata = (scientific, common)
            if species_id in names and names[species_id] != metadata:
                raise ValueError(f"Conflicting names for species_id {species_id}")
            names[species_id] = metadata
            audio = (path.parent / Path(row["reference_audio"]).expanduser()).resolve()
            validate_audio(audio)
            content_hash = file_hash(audio)
            pair = species_id, content_hash
            if pair in pairs:
                raise ValueError(f"Duplicate species/reference pair: {species_id}")
            pairs.add(pair)
            entries.append(SpeciesInput(species_id, scientific, common, audio, content_hash))
        except (ValueError, OSError) as exc:
            raise ValueError(f"species entry {index}: {exc}") from exc
    manifest = SpeciesManifest(path, digest, tuple(entries))
    verify_inputs(manifest)
    return manifest


def verify_inputs(manifest: SpeciesManifest) -> None:
    """Reject edited inputs instead of executing a different plan than reviewed."""
    if file_hash(manifest.path) != manifest.content_hash:
        raise ValueError("Manifest changed after review; select and review it again")
    for entry in manifest.entries:
        if file_hash(entry.audio) != entry.content_hash:
            raise ValueError(f"Reference changed after review: {entry.audio}")


def _selected_inputs(config) -> dict:
    """Preserve the corpus configuration; refuse to discard local CSV labels."""
    path = config.artifacts / "latest.json"
    latest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if latest.get("snapshot"):
        metadata = json.loads((Path(latest["snapshot"]) / "snapshot.json").read_text())
        if metadata.get("label_count", 0) and metadata.get("source") != "bigquery":
            raise ValueError(
                "Selected labels are local CSV data. Complete the CSV seed workflow first."
            )
    return latest


def _prepare(config, manifest, directory, backend):
    """Freeze reviewed audio, then reuse the normal recording/vector preparation."""
    prepared = []
    waveforms = {}
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        for entry in manifest.entries:
            path = directory / "input" / f"{entry.content_hash}{entry.audio.suffix.lower()}"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry.audio, path)
            if file_hash(path) != entry.content_hash:
                raise ValueError(f"Reference changed while copying: {entry.audio}")
            # Validate the frozen bytes too, before a decoder could pad a short clip.
            validate_audio(path)
            recording = identify_recording(path)
            frames = backend.prepare_audio(path)
            if frames.shape != (1, 160_000) or not np.isfinite(frames).all() or not np.any(frames):
                raise ValueError(
                    f"Reference must produce one finite, nonzero model frame: {entry.audio}"
                )
            # Different encodings can contain the same audio despite different file hashes.
            previous = waveforms.setdefault(entry.species_id, [])
            for previous_path, waveform in previous:
                if np.allclose(frames[0], waveform, rtol=0, atol=AUDIO_MATCH_TOLERANCE):
                    raise ValueError(
                        f"Duplicate prepared reference audio for {entry.species_id}: "
                        f"{entry.audio} matches {previous_path}. Keep one reference."
                    )
            previous.append((entry.audio, frames[0]))
            catalog.add_recording(recording)
            _store_recording(catalog, recording, backend, config.artifacts, frames=frames)
            window = recording_windows(recording)[0]
            waveform, _ = sf.read(catalog.audio_path(window.window_id), dtype="float32")
            if not np.array_equal(waveform, frames[0]):
                raise ValueError("Cached reference waveform differs from current preprocessing")
            prepared.append((entry, recording, window, catalog.audio_path(window.window_id)))
    return prepared


def _match_species(entries, existing):
    """Reuse matching catalog metadata; never rename species or infer another ID."""
    result = {}
    for entry in entries:
        matches = [
            row
            for row in existing
            if row["species_id"] == entry.species_id
            or normalize_name(row["scientific_name"]) == entry.scientific_name
        ]
        if len(matches) > 1:
            raise ValueError(f"Duplicate/colliding existing species: {entry.species_id}")
        desired = {
            "species_id": entry.species_id,
            "scientific_name": entry.scientific_name,
            "common_name": entry.common_name,
        }
        if matches:
            row = matches[0]
            if (
                row["species_id"] != entry.species_id
                or normalize_name(row["scientific_name"]) != entry.scientific_name
                or " ".join(row["common_name"].split()) != entry.common_name
            ):
                raise ValueError(f"Existing species metadata conflicts: {entry.species_id}")
            desired = {key: row[key] for key in desired}
        result[entry.species_id] = desired
    return list(result.values())


def _reference_rows(prepared, existing, storage, backend, directory):
    """Verify existing audio by waveform; preserve matching reference IDs and URIs."""
    result = []
    for entry, _, window, audio_path in prepared:
        reference_id = str(uuid5(REFERENCE_NAMESPACE, f"{entry.species_id}:{window.window_id}"))
        waveform, _ = sf.read(audio_path, dtype="float32")
        match = None
        for row in existing:
            if row["species_id"] != entry.species_id:
                continue
            path = directory / f"existing-{len(result)}{Path(row['audio_gcs_uri']).suffix}"
            storage.download_reference(row["audio_gcs_uri"], path)
            decoded, rate = sf.read(path, dtype="float32")
            if len(decoded) == 5 * rate:
                frames = backend.prepare_audio(path)
                if frames.shape == (1, 160_000) and np.allclose(
                    frames[0], waveform, rtol=0, atol=AUDIO_MATCH_TOLERANCE
                ):
                    if match is not None:
                        raise ValueError(f"Multiple existing references match: {entry.species_id}")
                    match = dict(row)
            if row["reference_id"] == reference_id and match != dict(row):
                raise ValueError(f"Reference ID conflicts with existing audio: {reference_id}")
        if match is None:
            match = {
                "reference_id": reference_id,
                "species_id": entry.species_id,
                "audio_gcs_uri": f"gs://{storage.config.bucket}/references/{entry.species_id}/{reference_id}.wav",
                "spectrogram_gcs_uri": None,
            }
        result.append(
            {
                key: match[key]
                for key in ("reference_id", "species_id", "audio_gcs_uri", "spectrogram_gcs_uri")
            }
        )
    if len({row["reference_id"] for row in result}) != len(result):
        raise ValueError("Multiple inputs resolve to the same reference")
    return result


def _metadata(prepared, species_rows, references, config):
    """Build add-only rows with stable species/window seed and submission IDs."""
    recordings, windows, annotations = {}, {}, []
    for entry, recording, window, _ in prepared:
        recordings[recording.recording_id] = {"recording_id": recording.recording_id}
        windows[window.window_id] = {
            "window_id": window.window_id,
            "recording_id": recording.recording_id,
            "start_seconds": window.start_seconds,
            "end_seconds": window.end_seconds,
        }
        key = f"new-species:{entry.species_id}:{window.window_id}"
        annotations.append(
            {
                "annotation_id": str(uuid5(SEED_NAMESPACE, key)),
                "window_id": window.window_id,
                "candidate_id": None,
                "query_species_id": entry.species_id,
                "response": "present",
                "annotator_id": "new-species",
                "annotator_name": "new-species",
                "submission_id": str(uuid5(SUBMISSION_NAMESPACE, key)),
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
    return {
        "species": species_rows,
        "reference_media": references,
        "recordings": list(recordings.values()),
        "windows": list(windows.values()),
        "annotations": annotations,
    }


def add_species(config, manifest: SpeciesManifest, *, backend=None, bq=None, storage=None) -> Path:
    """Prepare references, commit one cloud batch, then refresh all annotations.

    - Always writes seed/reference data, independently of candidate publication.
    - Retain local artifacts and a receipt after failures; rerun the same manifest.
    - Shared cloud lock covers preflight, uploads, commit, and local registration.
    """
    verify_inputs(manifest)
    latest = _selected_inputs(config)
    fresh_workspace = not latest.get("config_id")
    directory = config.artifacts / "species-imports" / new_run_id()
    receipt = {
        "stage": "validated",
        "manifest": str(manifest.path),
        "manifest_hash": manifest.content_hash,
        "inputs": [asdict(entry) for entry in manifest.entries],
        "dataset": f"{config.project}.{config.dataset}",
        "bucket": config.bucket,
    }
    # Path objects are represented as strings in the human-readable receipt.
    receipt["inputs"] = [{**row, "audio": str(row["audio"])} for row in receipt["inputs"]]

    def save(stage):
        receipt["stage"] = stage
        write_json(directory / "receipt.json", receipt)

    save("validated")
    try:
        if backend is None:
            from .bacpipe_adapter import BacpipePerch

            backend = BacpipePerch(config.artifacts, config.device)
        if latest.get("config_id") and latest["config_id"] != backend.config_id:
            raise ValueError(
                "Embedding configuration differs from the active corpus; import compatible corpus embeddings first"
            )
        receipt["config_id"] = backend.config_id
        logger.info("Preparing %d species references locally...", len(manifest.entries))
        prepared = _prepare(config, manifest, directory, backend)
        save("prepared")
        if bq is None or storage is None:
            from .publish import check_cloud_access

            bq, storage = check_cloud_access(config, bq=bq, storage=storage)
        with storage.publication_lock():
            _selected_inputs(config)
            verify_inputs(manifest)
            existing_species, existing_references = read_catalog(bq)
            species_rows = _match_species(manifest.entries, existing_species)
            references = _reference_rows(prepared, existing_references, storage, backend, directory)
            tables = _metadata(prepared, species_rows, references, config)
            receipt["rows"] = tables
            save("uploading")
            for index, ((entry, _, window, audio), reference) in enumerate(
                zip(prepared, references, strict=True), 1
            ):
                logger.info(
                    "Uploading reference audio %d/%d: %s",
                    index,
                    len(prepared),
                    entry.species_id,
                )
                if not any(
                    row["reference_id"] == reference["reference_id"] for row in existing_references
                ):
                    storage.upload_reference(entry.species_id, reference["reference_id"], audio)
            insert_batch(bq, tables)
            save("metadata_committed")
            with Catalog(config.artifacts / "catalog.sqlite") as catalog:
                for row in tables["windows"]:
                    catalog.mark_registered(row["window_id"])
        # Select the configuration only after the complete cloud batch is committed.
        latest = _selected_inputs(config)
        latest["config_id"] = backend.config_id
        write_json(config.artifacts / "latest.json", latest)
        snapshot = refresh_annotations(config, bq)
        receipt["snapshot"] = str(snapshot)
        save("refreshed")
        save("complete")
        logger.info("Species import complete: %s", directory)
        if fresh_workspace:
            logger.info("Import corpus audio before generating useful similarity candidates")
        return snapshot
    except Exception as exc:
        receipt["error"] = str(exc)
        write_json(directory / "receipt.json", receipt)
        raise
