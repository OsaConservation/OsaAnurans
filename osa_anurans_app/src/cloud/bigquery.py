"""BigQuery reads, metadata insertion, and atomic candidate replacement."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from ..config import AppConfig
from ..progress import activity
from ..records import RecordingRecord, WindowRecord

CANDIDATE_COLUMNS = (
    "candidate_id",
    "window_id",
    "query_species_id",
    "created_at",
    "score",
    "selection_method",
)


MEDIA_COLUMNS = (
    "recording_id",
    "recording_start_seconds",
    "recording_end_seconds",
    "audio_gcs_uri",
    "spectrogram_gcs_uri",
)


def validate_candidates(path: Path, expected_count: int, *, published: bool = False) -> None:
    """Validate a local candidate file before any table replacement.

    Checks:
    - Positive expected count and exactly that many JSONL rows.
    - Required fields, nonempty IDs, and timezone-aware creation timestamps.
    - Finite numeric scores between zero and one; reject bool values.

    Boundaries:
    - Read one line at a time; do not load the full file into memory.
    - Duplicate IDs/pairs and cloud references are checked in staging SQL later.
    """
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("Refusing to replace candidates with an empty result")
    count = 0
    with path.open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != set(
                    CANDIDATE_COLUMNS + MEDIA_COLUMNS if published else CANDIDATE_COLUMNS
                ):
                    raise ValueError("Unexpected candidate fields")
                for field in CANDIDATE_COLUMNS[:3]:
                    if not isinstance(row[field], str) or not row[field]:
                        raise ValueError(f"Missing {field}")
                score = row["score"]
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError("Score must be numeric")
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Score must be finite and between zero and one")
                if row["selection_method"] not in ("similarity", "uncertainty"):
                    raise ValueError("selection_method must be similarity or uncertainty")
                if published:
                    _validate_candidate_media(row)
                timestamp = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                if timestamp.utcoffset() is None:
                    raise ValueError("Candidate timestamp must include a timezone")
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid candidate on line {count}: {exc}") from exc
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} candidates, found {count}")


def _validate_candidate_media(row: dict) -> None:
    """Validate copied source bounds and media fields before staging."""
    start, end = row["recording_start_seconds"], row["recording_end_seconds"]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in (start, end)
    ):
        raise ValueError("Invalid candidate source bounds")
    if start < 0 or not math.isclose(end - start, 5, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Candidate source bounds must span five seconds")
    if not isinstance(row["recording_id"], str) or not row["recording_id"]:
        raise ValueError("Missing candidate recording ID")
    for field in ("audio_gcs_uri", "spectrogram_gcs_uri"):
        uri = row[field]
        if field == "spectrogram_gcs_uri" and uri is None:
            continue
        if not isinstance(uri, str) or not uri.startswith("gs://") or not uri[5:].partition("/")[2]:
            raise ValueError(f"Invalid {field}")


def _bigquery():
    """Import the BigQuery SDK only when cloud code needs it."""
    from google.cloud import bigquery

    return bigquery


class BigQueryStore:
    """Read annotations and write window metadata/candidate queues in BigQuery.

    Operations:
    - Read annotator answers and existing windows.
    - Insert missing recording/window rows after audio upload.
    - Replace candidates only after validating a separate staging table.

    Boundaries:
    - Preserve species, references, existing windows, and annotator answers.
    - Never upload embedding vectors.
    """

    def __init__(self, config: AppConfig, client=None):
        """Use a supplied test client or create one for the configured project/location."""
        self.config = config
        self.dataset = f"{config.project}.{config.dataset}"
        self.client = (
            client
            if client is not None
            else _bigquery().Client(project=config.project, location=config.location)
        )

    def _table(self, name: str) -> str:
        """Return a quoted table identifier within the configured dataset."""
        return f"`{self.dataset}.{name}`"

    def _query(self, sql: str, parameters=()):
        """Run parameterized SQL in the configured location and wait for results."""
        options = _bigquery().QueryJobConfig(query_parameters=list(parameters))
        with activity("BigQuery: waiting for query"):
            return self.client.query(
                sql, job_config=options, location=self.config.location
            ).result()

    def check_access(self) -> None:
        """Read-only checks; IAM write permissions are still enforced at writes."""
        self.client.get_dataset(self.dataset)
        for table in ("species", "annotations", "recordings", "windows", "candidates"):
            self.client.get_table(f"{self.dataset}.{table}")
        list(self._query("SELECT 1 AS accessible"))

    def fetch_annotations(self) -> list[dict]:
        """Read raw votes/provenance; training locates vectors by window ID locally."""
        return [dict(row) for row in self._query(f"SELECT * FROM {self._table('annotations')}")]

    def protected_window_ids(self) -> set[str]:
        """Protect current candidates only; annotations do not require cloud audio."""
        return {
            row["window_id"]
            for row in self._query(f"SELECT DISTINCT window_id FROM {self._table('candidates')}")
        }

    def reference_audio_uris(self) -> set[str]:
        """Protect reference copies, including any historical references in the window prefix."""
        return {
            row["audio_gcs_uri"]
            for row in self._query(f"SELECT audio_gcs_uri FROM {self._table('reference_media')}")
            if row["audio_gcs_uri"]
        }

    def storage_usage(self) -> dict[str, int]:
        """Read current logical table bytes from dataset metadata; do not scan rows.

        - Include all dataset tables, including temporary staging tables if present.
        - Metadata can lag writes; this is table data size, not query usage or a bill.
        """
        tables = self.client.list_tables(self.dataset)
        size = sum(self.client.get_table(table.reference).num_bytes or 0 for table in tables)
        return {"logical_bytes": size}

    def existing_windows(self, window_ids) -> dict[str, dict]:
        """Fetch metadata for requested IDs in batches of up to 5000.

        Rules:
        - Deduplicate requested IDs before querying.
        - Reject duplicate window IDs already present in BigQuery.
        - Return an ID-to-metadata dictionary; absent IDs need insertion.
        """
        ids = sorted(set(window_ids))
        found = {}
        for offset in range(0, len(ids), 5000):
            parameter = _bigquery().ArrayQueryParameter(
                "window_ids", "STRING", ids[offset : offset + 5000]
            )
            rows = self._query(
                f"SELECT * FROM {self._table('windows')} WHERE window_id IN UNNEST(@window_ids)",
                [parameter],
            )
            for row in rows:
                key = row["window_id"]
                if key in found:
                    raise ValueError(f"Duplicate window ID in BigQuery: {key}")
                found[key] = dict(row)
        return found

    def _load_rows(self, rows: list[dict], name: str, schema) -> str:
        """Load JSON metadata into a staging table, set expiry, and return its name."""
        destination = f"{self.dataset}.{name}"
        config = _bigquery().LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
        with activity(f"BigQuery: uploading {len(rows)} metadata rows"):
            self.client.load_table_from_json(
                rows, destination, job_config=config, location=self.config.location
            ).result()
        self._expire_table(destination)
        return destination

    def _expire_table(self, destination: str) -> None:
        """Set a one-day expiry so abandoned staging tables do not remain forever."""
        table = self.client.get_table(destination)
        table.expires = datetime.now(UTC) + timedelta(days=1)
        self.client.update_table(table, ["expires"])

    def upsert_windows(
        self, recordings: list[RecordingRecord], windows: list[WindowRecord]
    ) -> None:
        """Insert missing recording/window rows without replacing existing metadata.

        Steps:
        1. Validate unique IDs, source bounds, and recording metadata.
        2. Load the batch into a temporary staging table.
        3. Reject conflicts with existing window metadata inside a transaction.
        4. MERGE missing recording and window rows; preserve existing rows.
        5. Delete the staging table after the query, including on query failure.

        Boundaries:
        - Window registration is independent of media upload.
        - SQL avoids normal retry duplicates; BigQuery has no enforced unique key.
        """
        if not windows:
            return
        recording_ids = {recording.recording_id for recording in recordings}
        if len({window.window_id for window in windows}) != len(windows):
            raise ValueError("Duplicate window IDs in metadata upload")
        for window in windows:
            if window.recording_id not in recording_ids:
                raise ValueError(f"Recording metadata missing for {window.window_id}")
            if window.sample_rate <= 0 or window.end_sample <= window.start_sample:
                raise ValueError(f"Invalid window bounds: {window.window_id}")

        api = _bigquery()
        token = uuid4().hex
        stage = f"_window_upload_{token}"
        schema = [
            api.SchemaField("window_id", "STRING", mode="REQUIRED"),
            api.SchemaField("recording_id", "STRING", mode="REQUIRED"),
            api.SchemaField("start_seconds", "FLOAT64", mode="REQUIRED"),
            api.SchemaField("end_seconds", "FLOAT64", mode="REQUIRED"),
        ]
        rows = [
            {
                "window_id": window.window_id,
                "recording_id": window.recording_id,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
            }
            for window in windows
        ]
        destination = self._load_rows(rows, stage, schema)
        try:
            self._query(f"""
                BEGIN TRANSACTION;
                ASSERT NOT EXISTS (
                    SELECT 1 FROM {self._table(stage)} AS s
                    JOIN {self._table("windows")} AS t USING (window_id)
                    WHERE t.recording_id != s.recording_id
                       OR ABS(t.start_seconds - s.start_seconds) > 0.000001
                       OR ABS(t.end_seconds - s.end_seconds) > 0.000001
                ) AS 'Existing window metadata conflicts with local window';
                MERGE {self._table("recordings")} AS t
                USING (SELECT DISTINCT recording_id FROM {self._table(stage)}) AS s
                ON t.recording_id = s.recording_id
                WHEN NOT MATCHED THEN INSERT (recording_id) VALUES (s.recording_id);
                MERGE {self._table("windows")} AS t
                USING {self._table(stage)} AS s ON t.window_id = s.window_id
                WHEN NOT MATCHED THEN INSERT
                    (window_id, recording_id, start_seconds, end_seconds)
                VALUES
                    (s.window_id, s.recording_id, s.start_seconds, s.end_seconds);
                COMMIT TRANSACTION;
            """)
        finally:
            self.client.delete_table(destination, not_found_ok=True)

    def replace_candidates(self, path: Path, expected_count: int, run_id: str) -> None:
        """Replace the annotation candidate queue from one saved JSONL selection.

        Steps:
        1. Validate the local file and load a uniquely named staging table.
        2. Set a one-day expiry for failed/abandoned staging data.
        3. In one transaction, check count, unique IDs/pairs, source metadata, and species.
        4. Truncate candidates and insert the validated rows, then commit.
        5. Delete staging after success; retain failed query staging for diagnosis.

        Safety:
        - Validation/query failure before commit preserves the previous queue.
        - Existing annotations and windows remain unchanged.
        """
        validate_candidates(path, expected_count, published=True)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                expected_uri = f"gs://{self.config.bucket}/{self.config.prefix.rstrip('/')}/{row['window_id']}.wav"
                if row["audio_gcs_uri"] != expected_uri:
                    raise ValueError(
                        "Candidate audio must use its configured canonical window path"
                    )
                spectrogram = row["spectrogram_gcs_uri"]
                if spectrogram is not None and not spectrogram.startswith(
                    f"gs://{self.config.bucket}/"
                ):
                    raise ValueError("Candidate spectrogram must use the configured bucket")
        api = _bigquery()
        if not run_id:
            raise ValueError("A publication run ID is required")
        # Each attempt owns its staging table, even when the same run is retried.
        stage = f"_candidate_upload_{uuid4().hex}"
        destination = f"{self.dataset}.{stage}"
        field_types = {
            "score": "FLOAT64",
            "created_at": "TIMESTAMP",
            "recording_start_seconds": "FLOAT64",
            "recording_end_seconds": "FLOAT64",
        }
        schema = [
            api.SchemaField(
                name,
                field_types.get(name, "STRING"),
                mode="NULLABLE" if name == "spectrogram_gcs_uri" else "REQUIRED",
            )
            for name in CANDIDATE_COLUMNS + MEDIA_COLUMNS
        ]
        options = api.LoadJobConfig(
            schema=schema,
            source_format=api.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition="WRITE_TRUNCATE",
        )
        with (
            activity(f"BigQuery: uploading {expected_count} candidates"),
            path.open("rb") as stream,
        ):
            self.client.load_table_from_file(
                stream,
                destination,
                job_config=options,
                location=self.config.location,
            ).result()
        self._expire_table(destination)
        parameter = api.ScalarQueryParameter("expected_count", "INT64", expected_count)
        # Assertions and replacement share one transaction and snapshot.
        self._query(
            f"""
            BEGIN TRANSACTION;
            ASSERT (SELECT COUNT(*) FROM {self._table(stage)}) = @expected_count
                AS 'Candidate count mismatch';
            ASSERT (SELECT COUNT(DISTINCT candidate_id) FROM {self._table(stage)})
                = @expected_count AS 'Duplicate candidate IDs';
            ASSERT (SELECT COUNT(*) FROM (
                SELECT window_id, query_species_id FROM {self._table(stage)}
                GROUP BY window_id, query_species_id
            )) = @expected_count AS 'Duplicate window/species candidates';
            ASSERT NOT EXISTS (
                SELECT 1 FROM {self._table(stage)} AS s
                LEFT JOIN {self._table("windows")} AS w USING (window_id)
                WHERE w.window_id IS NULL OR w.recording_id != s.recording_id
                   OR ABS(w.start_seconds - s.recording_start_seconds) > 0.000001
                   OR ABS(w.end_seconds - s.recording_end_seconds) > 0.000001
            ) AS 'Candidate window metadata missing or conflicting';
            ASSERT NOT EXISTS (
                SELECT 1 FROM {self._table(stage)} AS s
                LEFT JOIN {self._table("species")} AS p
                  ON s.query_species_id = p.species_id
                WHERE p.species_id IS NULL
            ) AS 'Candidate species missing';
            TRUNCATE TABLE {self._table("candidates")};
            INSERT INTO {self._table("candidates")}
                ({", ".join(CANDIDATE_COLUMNS + MEDIA_COLUMNS)})
            SELECT {", ".join(CANDIDATE_COLUMNS + MEDIA_COLUMNS)} FROM {self._table(stage)};
            COMMIT TRANSACTION;
        """,
            [parameter],
        )
        self.client.delete_table(destination, not_found_ok=True)
