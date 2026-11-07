"""Cloud adapters are exercised with in-memory clients; no credentials or writes."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.api_core.exceptions import PreconditionFailed
from osa_anurans_app.src.cloud.bigquery import BigQueryStore, validate_candidates
from osa_anurans_app.src.cloud.storage import StorageStore, file_checksum
from osa_anurans_app.src.config import AppConfig
from osa_anurans_app.src.publish import check_cloud_access, publish_candidates, sync_windows
from osa_anurans_app.src.records import RecordingRecord, WindowRecord


@pytest.fixture
def config(tmp_path):
    return AppConfig((tmp_path,), tmp_path / "annotations.csv", tmp_path / "artifacts")


def candidate(window_id="w", species="A", candidate_id="c"):
    return {
        "candidate_id": candidate_id,
        "window_id": window_id,
        "query_species_id": species,
        "created_at": "2026-11-06T12:00:00+00:00",
        "score": 0.2,
    }


def write_candidates(tmp_path, rows):
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class Job:
    def __init__(self, result=None, error=None):
        self.rows = [] if result is None else result
        self.error = error

    def result(self):
        if self.error:
            raise self.error
        return self.rows


class BigQueryClient:
    """Simulate load isolation and transactional replacement, not SQL execution."""

    def __init__(self):
        self.current = [candidate(candidate_id="old")]
        self.staging = []
        self.windows = {"w": {"window_id": "w", "audio_gcs_uri": "gs://bucket/w.wav"}}
        self.species = {"A", "B"}
        self.queries = []
        self.loads = 0
        self.load_error = None
        self.commit_error = None
        self.deleted = []

    def load_table_from_file(self, stream, destination, **kwargs):
        self.loads += 1
        self.staging = [json.loads(line) for line in stream]
        assert kwargs["job_config"].write_disposition == "WRITE_TRUNCATE"
        assert destination.endswith("candidates") is False
        return Job(error=self.load_error)

    def query(self, sql, **kwargs):
        self.queries.append(sql)
        if "TRUNCATE TABLE" not in sql:
            return Job()

        assert sql.index("BEGIN TRANSACTION") < sql.index("ASSERT")
        assert sql.index("ASSERT") < sql.index("TRUNCATE TABLE")
        assert sql.index("TRUNCATE TABLE") < sql.index("INSERT INTO")
        assert sql.index("INSERT INTO") < sql.index("COMMIT TRANSACTION")
        if self.commit_error:
            return Job(error=self.commit_error)
        ids = [row["candidate_id"] for row in self.staging]
        pairs = [(row["window_id"], row["query_species_id"]) for row in self.staging]
        if len(set(ids)) != len(ids) or len(set(pairs)) != len(pairs):
            return Job(error=ValueError("Duplicate staging rows"))
        if any(row["window_id"] not in self.windows for row in self.staging):
            return Job(error=ValueError("Unknown window"))
        if any(row["query_species_id"] not in self.species for row in self.staging):
            return Job(error=ValueError("Unknown species"))
        self.current = list(self.staging)
        return Job()

    def get_table(self, name):
        return SimpleNamespace(table_id=name, expires=None)

    def update_table(self, table, fields):
        assert table.expires is not None

    def delete_table(self, name, **kwargs):
        self.deleted.append(name)


def test_candidate_load_failure_never_replaces_existing(config, tmp_path):
    client = BigQueryClient()
    client.load_error = RuntimeError("load failed")
    path = write_candidates(tmp_path, [candidate()])
    with pytest.raises(RuntimeError, match="load failed"):
        BigQueryStore(config, client).replace_candidates(path, 1, "run")
    assert client.current[0]["candidate_id"] == "old"
    assert not client.queries


@pytest.mark.parametrize(
    "rows",
    [
        [candidate(), candidate()],
        [candidate(window_id="missing")],
        [candidate(species="missing")],
    ],
)
def test_staging_validation_failure_preserves_current_candidates(config, tmp_path, rows):
    client = BigQueryClient()
    path = write_candidates(tmp_path, rows)
    with pytest.raises(ValueError):
        BigQueryStore(config, client).replace_candidates(path, len(rows), "run")
    assert client.current[0]["candidate_id"] == "old"
    assert not client.deleted  # Failed staging is retained for diagnosis, with TTL.


def test_successful_replacement_and_commit_failure(config, tmp_path):
    client = BigQueryClient()
    store = BigQueryStore(config, client)
    rows = [candidate(), candidate(species="B", candidate_id="c2")]
    path = write_candidates(tmp_path, rows)
    store.replace_candidates(path, 2, "run")
    assert client.current == rows
    assert client.deleted
    client.commit_error = RuntimeError("transaction aborted")
    write_candidates(tmp_path, [candidate(candidate_id="new")])
    with pytest.raises(RuntimeError, match="transaction aborted"):
        store.replace_candidates(path, 1, "next-run")
    assert client.current == rows


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(score=float("nan")),
        lambda row: row.update(score=True),
        lambda row: row.update(score=1.1),
        lambda row: row.update(created_at="2026-11-06T12:00:00"),
        lambda row: row.update(extra="unexpected"),
        lambda row: row.update(window_id=""),
    ],
)
def test_invalid_candidates_rejected_before_cloud_access(config, tmp_path, mutate):
    row = candidate()
    mutate(row)
    client = BigQueryClient()
    with pytest.raises(ValueError, match="Invalid candidate"):
        BigQueryStore(config, client).replace_candidates(
            write_candidates(tmp_path, [row]), 1, "run"
        )
    assert client.loads == 0


def test_empty_or_incomplete_candidates_never_truncate(tmp_path):
    path = write_candidates(tmp_path, [])
    with pytest.raises(ValueError, match="empty"):
        validate_candidates(path, 0)
    with pytest.raises(ValueError, match="Expected 1"):
        validate_candidates(path, 1)


class Blob:
    def __init__(self):
        self.data = None
        self.calls = []

    def upload_from_filename(self, filename, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["if_generation_match"] == 0
        if self.data is not None:
            raise PreconditionFailed("already exists")
        self.data = Path(filename).read_bytes()
        self.size = len(self.data)
        self.crc32c = file_checksum(Path(filename))

    def reload(self):
        pass


class StorageClient:
    def __init__(self):
        self.blobs = {}

    def bucket(self, name):
        return self

    def blob(self, name):
        return self.blobs.setdefault(name, Blob())


def test_create_only_audio_upload_reuses_only_identical_content(config, tmp_path):
    path = tmp_path / "w.wav"
    path.write_bytes(b"window audio")
    client = StorageClient()
    store = StorageStore(config, client)
    window = WindowRecord("w", "r", 0, 5, 1)
    uri = store.upload_window(window, path)
    assert uri == "gs://osa-frog-data/windows/v1/w.wav"
    assert store.upload_window(window, path) == uri
    path.write_bytes(b"different audio")
    with pytest.raises(ValueError, match="differs"):
        store.upload_window(window, path)
    assert client.blobs["windows/v1/w.wav"].data == b"window audio"


class Catalog:
    def __init__(self, path):
        self.window = WindowRecord("w", "r", 0, 5, 1)
        self.path = path
        self.uploaded = []
        self.registered = set()

    def pending_windows(self):
        return [] if "w" in self.registered else [self.window]

    def windows(self):
        return [self.window]

    def stored_recordings(self):
        return [RecordingRecord("r", self.path, "hash", 1, 5)]

    def audio_path(self, window_id):
        return self.path

    def mark_uploaded(self, window_id, uri):
        self.uploaded.append((window_id, uri))
        self.window = replace(self.window, audio_gcs_uri=uri)

    def mark_registered(self, window_id):
        self.registered.add(window_id)


class MetadataStore:
    def __init__(self):
        self.fail = True
        self.rows = []
        self.published = False

    def existing_windows(self, ids):
        return {}

    def upsert_windows(self, recordings, windows):
        assert all(window.audio_gcs_uri for window in windows)
        if self.fail:
            raise RuntimeError("metadata unavailable")
        self.rows.extend(windows)

    def replace_candidates(self, path, expected_count, run_id):
        self.published = True


def test_sync_resumes_upload_after_metadata_failure_independent_of_training(config, tmp_path):
    path = tmp_path / "w.wav"
    path.write_bytes(b"window audio")
    catalog = Catalog(path)
    bq = MetadataStore()
    storage_client = StorageClient()
    storage = StorageStore(config, storage_client)
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        sync_windows(config, catalog, bq, storage, window_ids={"w"})
    assert catalog.uploaded
    assert not catalog.registered
    bq.fail = False
    assert sync_windows(config, catalog, bq, storage, window_ids={"w"}) == 1
    assert sync_windows(config, catalog, bq, storage, window_ids={"w"}) == 0
    assert catalog.registered == {"w"}
    assert len(bq.rows) == 1


def test_failed_sync_prevents_candidate_publication(config, tmp_path):
    path = tmp_path / "w.wav"
    path.write_bytes(b"window audio")
    bq = MetadataStore()
    candidates = write_candidates(tmp_path, [candidate()])
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        publish_candidates(
            config, Catalog(path), candidates, 1, "run", bq, StorageStore(config, StorageClient())
        )
    assert not bq.published
    assert json.loads((tmp_path / "publish.json").read_text())["state"] == "syncing"


def test_auth_failure_is_actionable_and_does_not_hide_cause(config):
    class Denied:
        def check_access(self):
            raise PermissionError("denied")

    with pytest.raises(RuntimeError, match="application-default login") as error:
        check_cloud_access(config, Denied(), Denied())
    assert isinstance(error.value.__cause__, PermissionError)


def test_publication_uploads_only_candidate_windows_and_reuses_shared_audio(config, tmp_path):
    class ExtraWindowsCatalog(Catalog):
        def windows(self):
            return [self.window, WindowRecord("unused", "r", 5, 10, 1)]

        def pending_windows(self):
            return [window for window in self.windows() if window.window_id not in self.registered]

    audio = tmp_path / "w.wav"
    audio.write_bytes(b"window audio")
    catalog = ExtraWindowsCatalog(audio)
    bq = MetadataStore()
    bq.fail = False
    storage_client = StorageClient()
    candidates = write_candidates(
        tmp_path, [candidate(), candidate(species="B", candidate_id="c2")]
    )
    for _ in range(2):
        publish_candidates(
            config, catalog, candidates, 2, "run", bq, StorageStore(config, storage_client)
        )
    assert bq.published
    assert catalog.registered == {"w"}
    assert [row.window_id for row in bq.rows] == ["w"]
    assert set(storage_client.blobs) == {"windows/v1/w.wav"}
    assert len(storage_client.blobs["windows/v1/w.wav"].calls) == 1


def test_unknown_candidate_window_stops_before_upload_or_publication(config, tmp_path):
    audio = tmp_path / "w.wav"
    audio.write_bytes(b"window audio")
    bq = MetadataStore()
    bq.fail = False
    storage_client = StorageClient()
    candidates = write_candidates(tmp_path, [candidate(window_id="unknown")])
    with pytest.raises(ValueError, match="missing from local catalog"):
        publish_candidates(
            config, Catalog(audio), candidates, 1, "run", bq, StorageStore(config, storage_client)
        )
    assert not storage_client.blobs
    assert not bq.rows
    assert not bq.published


def test_publication_validates_candidates_before_upload(config, tmp_path):
    row = candidate()
    row["score"] = float("nan")
    path = write_candidates(tmp_path, [row])
    bq = MetadataStore()
    storage_client = StorageClient()
    with pytest.raises(ValueError, match="Invalid candidate"):
        publish_candidates(
            config,
            Catalog(tmp_path / "w.wav"),
            path,
            1,
            "run",
            bq,
            StorageStore(config, storage_client),
        )
    assert not storage_client.blobs
    assert not bq.published


def test_publication_still_checks_bucket_access(config):
    class BigQuery:
        def check_access(self):
            pass

    class DeniedStorage:
        def check_access(self):
            raise PermissionError("bucket denied")

    with pytest.raises(RuntimeError, match="bucket IAM") as caught:
        check_cloud_access(config, BigQuery(), DeniedStorage())
    assert isinstance(caught.value.__cause__, PermissionError)
