"""Upload window WAVs to GCS and write window/candidate rows to BigQuery.

Operations:
- sync_windows: Upload missing WAVs and insert missing recording/window rows.
- publish_candidates: Upload WAVs, insert window metadata, then replace candidates.
- cleanup_cloud_audio: Delete unused cloud clips while preserving local data.
- check_cloud_access: Check credentials and read access before cloud operations.
"""

import json
import logging
from pathlib import Path
from uuid import UUID

from .cloud.storage import file_checksum
from .config import AppConfig
from .files import write_json
from .progress import activity

logger = logging.getLogger(__name__)


def check_cloud_access(config: AppConfig, bq=None, storage=None) -> tuple:
    """Create cloud clients and check read access before uploads/table writes.

    Steps:
    1. Use supplied clients or create clients with Application Default Credentials.
    2. Check the BigQuery dataset/tables and GCS bucket metadata.

    Returns:
    - BigQueryStore and StorageStore instances.

    Unhappy paths:
    - Report credential/IAM failures while preserving the original exception.
    - Successful reads do not prove write permissions.
    """
    from .cloud.bigquery import BigQueryStore
    from .cloud.storage import StorageStore

    try:
        logger.info("Checking BigQuery and GCS access...")
        bq = bq or BigQueryStore(config)
        storage = storage or StorageStore(config)
        bq.check_access()
        storage.check_access()
    except Exception as exc:
        raise RuntimeError(
            "Cloud access check failed. Run 'gcloud auth application-default login' "
            "if credentials are missing/expired, and check BigQuery and bucket IAM "
            "permissions. This check does not grant or verify write permissions."
        ) from exc
    return bq, storage


def _same_window(window, cloud) -> bool:
    """Compare recording ID and source-second bounds with 1e-6 tolerance.

    Notes:
    - The caller checks the GCS audio URI separately after upload verification.
    """
    return (
        cloud["recording_id"] == window.recording_id
        and abs(cloud["start_seconds"] - window.start_seconds) < 0.000001
        and abs(cloud["end_seconds"] - window.end_seconds) < 0.000001
    )


def sync_windows(config: AppConfig, catalog, bq, storage, *, window_ids: set[str]) -> int:
    """Upload missing selected clips and metadata under the shared publication lock."""
    with storage.publication_lock():
        return _sync_windows(config, catalog, bq, storage, window_ids=window_ids)


def _sync_windows(config, catalog, bq, storage, *, window_ids, clips=None):
    """Verify selected audio separately from recording/window metadata registration.

    - Caller holds the publication lock.
    - Bulk GCS metadata avoids an existence request for every clip.
    - Registered metadata never implies the clip still exists.
    """
    windows = {window.window_id: window for window in catalog.windows()}
    missing = window_ids - windows.keys()
    if missing:
        raise ValueError(f"Candidate windows missing from local catalog: {sorted(missing)}")
    clips = storage.list_window_audio() if clips is None else clips
    pending_ids = {window.window_id for window in catalog.pending_windows()}
    pending = [windows[key] for key in sorted(window_ids) if key in pending_ids]
    existing = bq.existing_windows(window_ids) if pending else {}
    changed = set()
    with activity(
        "GCS: uploading/verifying selected clips", total=len(window_ids), unit="clip"
    ) as progress:
        for key in sorted(window_ids):
            window = windows[key]
            if key in existing and not _same_window(window, existing[key]):
                raise ValueError(f"BigQuery window mapping conflicts: {key}")
            path = catalog.audio_path(key)
            if path is None or not path.is_file():
                raise ValueError(f"Local audio missing for window {key}")
            blob = clips.get(key)
            if blob is None:
                storage.upload_window(window, path)
                changed.add(key)
            elif blob.size != path.stat().st_size or blob.crc32c != file_checksum(path):
                raise ValueError(f"Existing GCS audio differs from local window: {key}")
            progress.update(1)
    # Cloud clips are ready. A failed metadata write leaves registration pending.
    for start in range(0, len(pending), 500):
        batch = pending[start : start + 500]
        logger.info("BigQuery: inserting metadata for %s selected windows", len(batch))
        bq.upsert_windows(catalog.stored_recordings(), batch)
        for window in batch:
            catalog.mark_registered(window.window_id)
            changed.add(window.window_id)
    return len(changed)


def _candidate_upload(path, catalog, config):
    """Copy selection rows and add publication media/source fields; keep the saved run intact."""
    windows = {window.window_id: window for window in catalog.windows()}
    destination = path.parent / "candidate-upload.jsonl"
    temporary = destination.with_suffix(".tmp")
    try:
        with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for line in source:
                row = json.loads(line)
                window = windows[row["window_id"]]
                row.update(
                    recording_id=window.recording_id,
                    recording_start_seconds=window.start_seconds,
                    recording_end_seconds=window.end_seconds,
                    audio_gcs_uri=f"gs://{config.bucket}/{config.prefix.rstrip('/')}/{window.window_id}.wav",
                    spectrogram_gcs_uri=None,
                )
                target.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _restore_protected_audio(config, catalog, bq, storage, protected, clips):
    """Restore missing current candidate clips when local audio is available."""
    missing = protected - clips.keys()
    local_ids = {window.window_id for window in catalog.windows()}
    repairable = {
        window_id
        for window_id in missing & local_ids
        if (path := catalog.audio_path(window_id)) is not None and path.is_file()
    }
    if repairable:
        logger.info("Restoring %d missing protected clips from local audio", len(repairable))
        _sync_windows(config, catalog, bq, storage, window_ids=repairable, clips=clips)
    unavailable = missing - repairable
    if unavailable:
        logger.warning(
            "%d protected clips are missing in GCS and have no available local audio",
            len(unavailable),
        )


def _cleanup_unused_audio(config, catalog, bq, storage):
    """Delete canonical WAVs outside the current candidate queue.

    - Include GCS-only orphans from interrupted uploads.
    - Annotation-only audio is no longer needed for local embedding training.
    - No grace period, metadata deletion, or local file deletion.
    """
    logger.info("GCS: listing window clips for cleanup...")
    clips = storage.list_window_audio()
    protected = bq.protected_window_ids()
    _restore_protected_audio(config, catalog, bq, storage, protected, clips)
    reference_uris = bq.reference_audio_uris()
    unused = []
    for window_id, blob in clips.items():
        uri = f"gs://{config.bucket}/{blob.name}"
        if window_id in protected or uri in reference_uris:
            continue
        try:
            canonical = str(UUID(window_id)) == window_id
        except ValueError:
            canonical = False
        if canonical:
            unused.append((window_id, blob))
    removed = removed_bytes = 0
    with activity("GCS: removing unused window clips", total=len(unused), unit="clip") as progress:
        for window_id, blob in unused:
            if storage.delete_window_audio(window_id, blob):
                removed += 1
                removed_bytes += blob.size or 0
            progress.update(1)
    result = {"removed_clips": removed, "removed_bytes": removed_bytes}
    logger.info(
        "Cleanup complete: %d clips removed (%.3f GB)", removed, removed_bytes / 1_000_000_000
    )
    return result


def _report_storage_usage(bq, storage):
    """Show remaining storage; a metadata-read failure must not undo successful cleanup."""
    usage = {}
    for service, store in (("gcs", storage), ("bigquery", bq)):
        try:
            usage[service] = store.storage_usage()
        except Exception as exc:
            logger.warning("%s storage usage unavailable: %s", service, exc)
    if "gcs" in usage:
        active = usage["gcs"]["active_bytes"] / 1_000_000_000
        deleted = usage["gcs"]["soft_deleted_bytes"] / 1_000_000_000
        logger.info(
            "GCS storage: %.3f GB active + %.3f GB soft-deleted = %.3f GB retained",
            active,
            deleted,
            active + deleted,
        )
    if "bigquery" in usage:
        logger.info(
            "BigQuery storage: %.6f GB table data (metadata may lag recent writes)",
            usage["bigquery"]["logical_bytes"] / 1_000_000_000,
        )
    return usage


def cleanup_cloud_audio(config, catalog, bq, storage, *, publication_manifest=None):
    """Remove unused cloud WAVs; retry against current references, never republish.

    - Optional run manifest records that publication succeeded before cleanup.
    - Re-listing is enough to resume; already-missing clips are successful retries.
    """
    report = config.artifacts / "cloud-cleanup.json"
    publication = None
    # Acquire before writing reports so a second process cannot change active state.
    with storage.publication_lock():
        if publication_manifest is not None:
            report = publication_manifest.parent / "cleanup.json"
            publication = json.loads(publication_manifest.read_text(encoding="utf-8"))
        payload = {"state": "cleaning", "bucket": config.bucket, "prefix": config.prefix}
        write_json(report, payload)
        if publication is not None:
            publication["state"] = "cleaning"
            write_json(publication_manifest, publication)
        try:
            payload.update(_cleanup_unused_audio(config, catalog, bq, storage), state="complete")
        except Exception as exc:
            payload.update(state="cleanup_incomplete", error=str(exc))
            write_json(report, payload)
            if publication is not None:
                publication["state"] = "cleanup_incomplete"
                write_json(publication_manifest, publication)
            raise RuntimeError(
                f"Cloud audio cleanup incomplete: {exc}. "
                "Retry the cleanup action; the candidate queue was not changed by cleanup."
            ) from exc
        payload["storage_usage"] = _report_storage_usage(bq, storage)
        write_json(report, payload)
        if publication is not None:
            publication["state"] = "complete"
            publication["cleanup"] = payload
            write_json(publication_manifest, publication)
        return payload


def publish_candidates(
    config: AppConfig,
    catalog,
    candidates_path: Path,
    expected_count: int,
    run_id: str,
    bq,
    storage,
    *,
    cleanup: bool = False,
) -> bool:
    """Upload selected audio, then transactionally replace the candidate queue.

    - Share a GCS lock with cleanup, including during uploads and metadata writes.
    - Verify selected clips independently of metadata registration.
    - A run interrupted during cleanup never replaces the queue again.
    - Cleanup requires explicit opt-in, including on retries.
    - Return whether this call replaced the candidate queue.
    """
    from .cloud.bigquery import validate_candidates

    validate_candidates(candidates_path, expected_count)
    manifest = candidates_path.parent / "publish.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("state") in {"cleaning", "cleanup_incomplete"}:
            if cleanup:
                logger.info(
                    "Retrying cleanup against the current queue; candidates were already published"
                )
                cleanup_cloud_audio(config, catalog, bq, storage, publication_manifest=manifest)
            else:
                logger.info(
                    "Run was already published; select cleanup to finish removing unused clips"
                )
            return False
    payload = {
        "run_id": run_id,
        "candidate_count": expected_count,
        "candidates_path": str(candidates_path.resolve()),
        "state": "syncing",
    }
    with candidates_path.open(encoding="utf-8") as stream:
        window_ids = {json.loads(line)["window_id"] for line in stream}
    with storage.publication_lock():
        write_json(manifest, payload)
        logger.info("GCS: listing existing window clips...")
        clips = storage.list_window_audio()
        _sync_windows(config, catalog, bq, storage, window_ids=window_ids, clips=clips)
        payload["state"] = "replacing_candidates"
        write_json(manifest, payload)
        logger.info("BigQuery: replacing candidate queue with %s candidates", expected_count)
        upload_path = _candidate_upload(candidates_path, catalog, config)
        bq.replace_candidates(upload_path, expected_count, run_id)
        payload["state"] = "complete"
        write_json(manifest, payload)
    logger.info("Upload complete: %s candidates ready for annotation", expected_count)
    if cleanup:
        cleanup_cloud_audio(config, catalog, bq, storage, publication_manifest=manifest)
    return True
