"""One-time seed of CSV-derived presence labels after a full local import.

Usage:
- python -m osa_anurans_app.scripts.seed_grillos --config <config.toml>
- Run the normal import command with --skip-invalid first.

Writes:
- BigQuery recording/window metadata for annotated windows; no audio uploads.
- One present vote per window/species, attributed to grillos-seed-gt.

Rules:
- Seed only the completed import's CSV snapshot, never fetched annotator answers.
- Reuse window metadata and deterministic annotation IDs on retries.
- Preserve existing answers and candidates; do not upload embeddings.
- This script can be removed after seeding; training does not import it.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ..src.annotations import load_snapshot
from ..src.catalog import Catalog
from ..src.cli import configure_cloud_environment
from ..src.config import load_config
from ..src.files import write_json
from ..src.publish import check_cloud_access

logger = logging.getLogger(__name__)
ANNOTATOR = "grillos-seed-gt"
SUBMISSION = "grillos-seed-gt-v1"


def seed_rows(labels, created_at: str) -> list[dict]:
    """Build one stable present vote per CSV-derived window/species pair.

    Rules:
    - Reject non-presence snapshots and duplicate pairs before cloud writes.
    - No candidate ID is needed for imported ground truth.
    """
    rows = []
    seen = set()
    for label in labels:
        pair = (label.window_id, label.species_id)
        if pair in seen or label.present != 1 or label.absent != 0:
            raise ValueError("Expected unique CSV presence labels")
        seen.add(pair)
        rows.append(
            {
                "annotation_id": str(uuid5(NAMESPACE_URL, f"{SUBMISSION}:{pair[0]}:{pair[1]}")),
                "window_id": pair[0],
                "candidate_id": None,
                "query_species_id": pair[1],
                "response": "present",
                "annotator_id": ANNOTATOR,
                "annotator_name": ANNOTATOR,
                "submission_id": SUBMISSION,
                "created_at": created_at,
            }
        )
    if not rows:
        raise ValueError("No CSV presence labels to seed")
    return rows


def insert_annotations(bq, rows) -> None:
    """Stage and insert missing seed votes; preserve all existing answers.

    Checks:
    - Every seed references registered window metadata and an existing species.
    - Stable IDs may be reused only for matching votes.
    - Transactional MERGE makes completed retries insert no extra votes.
    """
    from uuid import uuid4

    stage = f"_grillos_seed_{uuid4().hex}"
    schema = bq.client.get_table(f"{bq.dataset}.annotations").schema
    destination = bq._load_rows(rows, stage, schema)
    try:
        bq._query(f"""
            BEGIN TRANSACTION;
            ASSERT NOT EXISTS (
                SELECT 1 FROM {bq._table(stage)} s
                LEFT JOIN {bq._table("windows")} w USING (window_id)
                LEFT JOIN {bq._table("species")} p ON p.species_id = s.query_species_id
                WHERE w.window_id IS NULL OR p.species_id IS NULL
            ) AS 'Seed references missing window or species';
            ASSERT NOT EXISTS (
                SELECT 1 FROM {bq._table(stage)} s
                JOIN {bq._table("annotations")} t USING (annotation_id)
                WHERE s.window_id != t.window_id OR s.query_species_id != t.query_species_id
                   OR s.response != t.response OR s.annotator_id != t.annotator_id
                   OR s.annotator_name != t.annotator_name
                   OR s.submission_id != t.submission_id OR t.candidate_id IS NOT NULL
            ) AS 'Existing seed annotation conflicts';
            MERGE {bq._table("annotations")} t
            USING {bq._table(stage)} s ON t.annotation_id = s.annotation_id
            WHEN NOT MATCHED THEN INSERT ROW;
            COMMIT TRANSACTION;
        """)
    finally:
        bq.client.delete_table(destination, not_found_ok=True)


def main() -> None:
    """Validate a completed import, register window metadata, then seed votes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if config.annotations_csv is None:
        raise ValueError("Seeding requires a separate config with annotations_csv set")
    configure_cloud_environment(config)
    latest = json.loads((config.artifacts / "latest.json").read_text())
    snapshot = Path(latest["snapshot"])
    # A fetched BigQuery snapshot must never be fed back as new seed votes.
    report = json.loads((snapshot.parents[1] / "import.json").read_text())
    metadata = json.loads((snapshot / "snapshot.json").read_text())
    if report["state"] != "complete" or metadata["source"] != str(config.annotations_csv):
        raise ValueError("Expected the completed CSV import snapshot")
    rows = seed_rows(load_snapshot(snapshot), datetime.now(UTC).isoformat())
    bq, storage = check_cloud_access(config)
    known_species = {
        row["species_id"] for row in bq._query(f"SELECT species_id FROM {bq._table('species')}")
    }
    if {row["query_species_id"] for row in rows} - known_species:
        raise ValueError("Seed contains species missing from BigQuery")
    window_ids = {row["window_id"] for row in rows}
    logger.info("Registering metadata for %d annotated windows", len(window_ids))
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        with storage.publication_lock():
            windows = [window for window in catalog.windows() if window.window_id in window_ids]
            bq.upsert_windows(catalog.stored_recordings(), windows)
            for window in windows:
                catalog.mark_registered(window.window_id)
    logger.info("Inserting %d ground-truth presence votes", len(rows))
    insert_annotations(bq, rows)
    write_json(
        snapshot / "seed.json",
        {
            "state": "complete",
            "annotations": len(rows),
            "windows": len(window_ids),
            "annotator_id": ANNOTATOR,
            "submission_id": SUBMISSION,
        },
    )
    logger.info("Seed complete; normal BigQuery refresh and training can now be used")


if __name__ == "__main__":
    main()
