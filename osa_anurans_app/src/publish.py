"""Upload window WAVs to GCS and write window/candidate rows to BigQuery.

Operations:
- sync_windows: Upload missing WAVs and insert missing recording/window rows.
- publish_candidates: Upload WAVs, insert window metadata, then replace candidates.
- cleanup_cloud_audio: Delete unused cloud clips while preserving local data.
- check_cloud_access: Check credentials and read access before cloud operations.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from uuid import UUID

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
    """Restore absent clips even when SQLite already marks their windows registered.

    - Caller holds the publication lock across all cloud writes.
    - One bulk GCS listing replaces individual existence requests.
    - BigQuery window rows and URIs remain valid when their clips are deleted.
    """
    windows = {window.window_id: window for window in catalog.windows()}
    missing = window_ids - windows.keys()
    if missing:
        raise ValueError(f"Candidate windows missing from local catalog: {sorted(missing)}")
    clips = storage.list_window_audio() if clips is None else clips
    pending_ids = {window.window_id for window in catalog.pending_windows()}
    pending = [
        windows[window_id]
        for window_id in sorted(window_ids)
        if window_id in pending_ids or window_id not in clips
    ]
    if not pending:
        logger.info("All %s selected clips exist in GCS and are registered", len(window_ids))
        return 0

    # Pending windows may belong to older content whose source path was replaced.
    recordings = catalog.stored_recordings()
    logger.info("Checking BigQuery metadata for %s pending window clips", len(pending))
    existing = bq.existing_windows(window.window_id for window in pending)
    count = 0
    # Bounded batches keep retries cheap without a BigQuery job per window.
    for start in range(0, len(pending), 500):
        batch = []
        selected = pending[start : start + 500]
        with activity(
            f"GCS: uploading/verifying clips {start + 1}-{start + len(selected)} of {len(pending)}",
            total=len(selected),
            unit="clip",
        ) as progress:
            for window in selected:
                cloud = existing.get(window.window_id)
                if cloud is not None and not _same_window(window, cloud):
                    raise ValueError(f"BigQuery window mapping conflicts: {window.window_id}")

                audio_path = catalog.audio_path(window.window_id)
                if audio_path is None:
                    raise ValueError(f"Local audio missing for window {window.window_id}")
                uri = storage.upload_window(window, audio_path)
                if cloud is not None and cloud.get("audio_gcs_uri") != uri:
                    raise ValueError(f"BigQuery window audio URI conflicts: {window.window_id}")
                catalog.mark_uploaded(window.window_id, uri)
                batch.append(replace(window, audio_gcs_uri=uri))
                progress.update(1)
        # WAV uploads are complete; now insert their missing BigQuery metadata.
        # If the query fails, leave the batch pending so a retry can reuse the WAVs.
        logger.info(
            "BigQuery: inserting metadata for %s clips (%s/%s processed)",
            len(batch),
            start,
            len(pending),
        )
        bq.upsert_windows(recordings, batch)
        for window in batch:
            catalog.mark_registered(window.window_id)
        count += len(batch)
        logger.info("Cloud clips ready: %s/%s", count, len(pending))
    return count


def _restore_protected_audio(config, catalog, bq, storage, protected, clips):
    """Repair protected clips deleted before a late annotation, when available locally."""
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
    """Delete canonical WAVs outside the current candidate/annotation snapshot.

    - Include GCS-only orphans from interrupted uploads.
    - A late annotation can race this snapshot; subsequent runs repair its audio.
    - No grace period, metadata deletion, or local file deletion.
    """
    logger.info("GCS: listing window clips for cleanup...")
    clips = storage.list_window_audio()
    protected = bq.protected_window_ids()
    _restore_protected_audio(config, catalog, bq, storage, protected, clips)
    # Refresh protection after uploads so annotations arriving during repair count too.
    protected = bq.protected_window_ids()
    unused = []
    for window_id, blob in clips.items():
        if window_id in protected:
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
    - Restore missing protected clips before publishing when local audio is available.
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
        protected = bq.protected_window_ids()
        # New selected windows are restored below; do not upload them twice.
        _restore_protected_audio(config, catalog, bq, storage, protected - window_ids, clips)
        _sync_windows(config, catalog, bq, storage, window_ids=window_ids, clips=clips)
        payload["state"] = "replacing_candidates"
        write_json(manifest, payload)
        logger.info("BigQuery: replacing candidate queue with %s candidates", expected_count)
        bq.replace_candidates(candidates_path, expected_count, run_id)
        payload["state"] = "complete"
        write_json(manifest, payload)
    logger.info("Upload complete: %s candidates ready for annotation", expected_count)
    if cleanup:
        cleanup_cloud_audio(config, catalog, bq, storage, publication_manifest=manifest)
    return True
