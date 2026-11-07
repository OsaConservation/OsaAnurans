"""Upload window WAVs to GCS and write window/candidate rows to BigQuery.

Operations:
- sync_windows: Upload missing WAVs and insert missing recording/window rows.
- publish_candidates: Upload WAVs, insert window metadata, then replace candidates.
- check_cloud_access: Check credentials and read access before cloud operations.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path

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
    """Upload pending window WAVs and insert their BigQuery metadata.

    Steps:
    1. Select requested window IDs not marked registered in SQLite.
    2. Fetch existing BigQuery windows by ID in batches and reject conflicts.
    3. Create GCS objects, or verify existing objects' size/checksum.
    4. Insert missing recording/window rows in BigQuery, 500 windows at a time.
    5. Mark each successful batch registered in SQLite.

    Returns:
    - Number of pending windows processed; zero if all are marked registered.

    Retry rules:
    - Existing GCS objects and BigQuery rows are reused when they match.
    - Failed batches remain pending for the next attempt.
    - Locally registered windows are skipped; external cloud deletion is not detected.
    """
    local_ids = {window.window_id for window in catalog.windows()}
    missing = window_ids - local_ids
    if missing:
        raise ValueError(f"Candidate windows missing from local catalog: {sorted(missing)}")
    pending = [window for window in catalog.pending_windows() if window.window_id in window_ids]
    if not pending:
        logger.info(
            "All %s selected window clips are already registered; no uploads needed",
            len(window_ids),
        )
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


def publish_candidates(
    config: AppConfig,
    catalog,
    candidates_path: Path,
    expected_count: int,
    run_id: str,
    bq,
    storage,
) -> None:
    """Upload selected audio/metadata, then replace the BigQuery candidate queue.

    Steps:
    1. Validate saved candidates and write publish.json progress state.
    2. Upload only their unique pending WAVs and insert missing window metadata.
    3. Replace candidates through BigQuery's staging/validation transaction.
    4. Mark publish.json complete only after replacement succeeds.

    Retry rules:
    - Reuse matching uploaded objects and locally registered windows.
    - Keep previous windows/annotations; replace only the candidate queue.
    - The manifest records the last attempted stage when an exception occurs.
    """
    from .cloud.bigquery import validate_candidates

    validate_candidates(candidates_path, expected_count)
    manifest = candidates_path.parent / "publish.json"
    payload = {
        "run_id": run_id,
        "candidate_count": expected_count,
        "candidates_path": str(candidates_path.resolve()),
        "state": "syncing",
    }
    write_json(manifest, payload)
    # Validation above precedes all uploads; shared windows need only one WAV.
    with candidates_path.open(encoding="utf-8") as stream:
        window_ids = {json.loads(line)["window_id"] for line in stream}
    sync_windows(config, catalog, bq, storage, window_ids=window_ids)
    payload["state"] = "replacing_candidates"
    write_json(manifest, payload)
    logger.info("BigQuery: replacing candidate queue with %s candidates", expected_count)
    bq.replace_candidates(candidates_path, expected_count, run_id)
    payload["state"] = "complete"
    write_json(manifest, payload)
    logger.info("Upload complete: %s candidates ready for annotation", expected_count)
