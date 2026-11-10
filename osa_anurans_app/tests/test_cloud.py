"""Cloud adapters are exercised with in-memory clients; no credentials or writes."""

import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed
from osa_anurans_app.src.cloud.bigquery import BigQueryStore, validate_candidates
from osa_anurans_app.src.cloud.storage import StorageStore, file_checksum
from osa_anurans_app.src.config import AppConfig
from osa_anurans_app.src.publish import (
    check_cloud_access,
    cleanup_cloud_audio,
    publish_candidates,
    sync_windows,
)
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
        "selection_method": "uncertainty",
    }


def upload_candidate(*args, **kwargs):
    row = candidate(*args, **kwargs)
    return dict(
        row,
        recording_id="r",
        recording_start_seconds=0,
        recording_end_seconds=5,
        audio_gcs_uri=f"gs://osa-frog-data/windows/v1/{row['window_id']}.wav",
        spectrogram_gcs_uri=None,
    )


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
    path = write_candidates(tmp_path, [upload_candidate()])
    with pytest.raises(RuntimeError, match="load failed"):
        BigQueryStore(config, client).replace_candidates(path, 1, "run")
    assert client.current[0]["candidate_id"] == "old"
    assert not client.queries


@pytest.mark.parametrize(
    "rows",
    [
        [upload_candidate(), upload_candidate()],
        [upload_candidate(window_id="missing")],
        [upload_candidate(species="missing")],
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
    rows = [
        upload_candidate(),
        dict(upload_candidate(species="B", candidate_id="c2"), selection_method="similarity"),
    ]
    path = write_candidates(tmp_path, rows)
    store.replace_candidates(path, 2, "run")
    assert client.current == rows
    assert client.deleted
    client.commit_error = RuntimeError("transaction aborted")
    write_candidates(tmp_path, [upload_candidate(candidate_id="new")])
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
        lambda row: row.update(selection_method="classifier"),
        lambda row: row.update(selection_method=None),
        lambda row: row.pop("selection_method"),
    ],
)
def test_invalid_candidates_rejected_before_cloud_access(config, tmp_path, mutate):
    row = upload_candidate()
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
    def __init__(self, client, name):
        self.client = client
        self.name = name
        self.bucket = SimpleNamespace(name=client.bucket_name)
        self.data = None
        self.calls = []
        self.size = 0
        self.generation = 0

    def upload_from_filename(self, filename, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["if_generation_match"] == 0
        if self.data is not None:
            raise PreconditionFailed("already exists")
        self.data = Path(filename).read_bytes()
        self.size = len(self.data)
        self.crc32c = file_checksum(Path(filename))
        self.client.generation += 1
        self.generation = self.client.generation

    def upload_from_string(self, text, **kwargs):
        assert kwargs["if_generation_match"] == 0
        if self.data is not None:
            raise PreconditionFailed("already exists")
        self.data = text.encode()
        self.size = len(self.data)
        self.client.generation += 1
        self.generation = self.client.generation

    def delete(self, *, if_generation_match):
        current = self.client.blobs.get(self.name)
        if current is None or current.data is None:
            raise NotFound("missing")
        if current.generation != if_generation_match:
            raise PreconditionFailed("generation changed")
        if self.name in self.client.delete_errors:
            raise self.client.delete_errors[self.name]
        self.client.soft_deleted.append(copy(current))
        del self.client.blobs[self.name]

    def reload(self):
        pass


class StorageClient:
    def __init__(self):
        self.blobs = {}
        self.generation = 0
        self.bucket_name = "osa-frog-data"
        self.delete_errors = {}
        self.list_calls = 0
        self.soft_deleted = []

    def bucket(self, name):
        self.bucket_name = name
        return self

    def blob(self, name):
        return self.blobs.setdefault(name, Blob(self, name))

    def list_blobs(self, bucket, *, prefix="", soft_deleted=False):
        self.list_calls += 1
        if soft_deleted:
            return self.soft_deleted
        return [
            copy(blob)
            for name, blob in self.blobs.items()
            if name.startswith(prefix) and blob.data is not None
        ]


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
        self.registered = set()

    def pending_windows(self):
        return [] if self.window.window_id in self.registered else [self.window]

    def windows(self):
        return [self.window]

    def stored_recordings(self):
        return [RecordingRecord("r", self.path, "hash", 1, 5)]

    def audio_path(self, window_id):
        return self.path

    def mark_registered(self, window_id):
        self.registered.add(window_id)


class MetadataStore:
    def __init__(self):
        self.fail = True
        self.rows = []
        self.published = False
        self.protected = set()
        self.references = set()

    def protected_window_ids(self):
        return self.protected

    def reference_audio_uris(self):
        return self.references

    def storage_usage(self):
        return {"logical_bytes": 1000}

    def existing_windows(self, ids):
        return {}

    def upsert_windows(self, recordings, windows):
        assert all(not hasattr(window, "audio_gcs_uri") for window in windows)
        if self.fail:
            raise RuntimeError("metadata unavailable")
        self.rows.extend(windows)

    def replace_candidates(self, path, expected_count, run_id):
        self.published = True
        self.protected = {json.loads(line)["window_id"] for line in path.read_text().splitlines()}


def test_sync_resumes_upload_after_metadata_failure_independent_of_training(config, tmp_path):
    path = tmp_path / "w.wav"
    path.write_bytes(b"window audio")
    catalog = Catalog(path)
    bq = MetadataStore()
    storage_client = StorageClient()
    storage = StorageStore(config, storage_client)
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        sync_windows(config, catalog, bq, storage, window_ids={"w"})
    assert storage_client.blobs["windows/v1/w.wav"].data == path.read_bytes()
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


CLIP_A = "11111111-1111-5111-8111-111111111111"
CLIP_B = "22222222-2222-5222-8222-222222222222"
CLIP_C = "33333333-3333-5333-8333-333333333333"


def add_clip(storage, tmp_path, window_id):
    path = tmp_path / f"{window_id}.wav"
    path.write_bytes(b"local window audio")
    storage.upload_window(WindowRecord(window_id, "r", 0, 5, 1), path)
    return path


def test_registered_clip_is_reuploaded_after_cloud_deletion(config, tmp_path):
    path = tmp_path / "w.wav"
    path.write_bytes(b"window audio")
    catalog = Catalog(path)
    bq = MetadataStore()
    bq.fail = False
    client = StorageClient()
    storage = StorageStore(config, client)
    sync_windows(config, catalog, bq, storage, window_ids={"w"})
    assert catalog.registered == {"w"}
    del client.blobs["windows/v1/w.wav"]
    before = client.list_calls
    assert sync_windows(config, catalog, bq, storage, window_ids={"w"}) == 1
    assert client.list_calls == before + 1
    assert client.blobs["windows/v1/w.wav"].data == b"window audio"


def test_cleanup_protects_referenced_clips_and_removes_gcs_only_orphans(config, tmp_path):
    client = StorageClient()
    storage = StorageStore(config, client)
    for window_id in [CLIP_A, CLIP_B, CLIP_C]:
        add_clip(storage, tmp_path, window_id)
    # Unexpected names and reference media must survive even without BQ references.
    for name in [
        f"reference_catalog/{CLIP_C}.wav",
        f"windows/v1/nested/{CLIP_C}.wav",
        "windows/v1/notes.wav",
    ]:
        client.blob(name).upload_from_string("keep", if_generation_match=0)
    bq = MetadataStore()
    bq.protected = {CLIP_A, CLIP_B}
    result = cleanup_cloud_audio(config, Catalog(tmp_path / "unused"), bq, storage)
    assert result["removed_clips"] == 1
    assert result["removed_bytes"] == len(b"local window audio")
    assert f"windows/v1/{CLIP_C}.wav" not in client.blobs
    assert len(client.blobs) == 5
    assert all((tmp_path / f"{window_id}.wav").exists() for window_id in [CLIP_A, CLIP_B, CLIP_C])
    assert bq.rows == []  # No metadata rows were removed or rewritten.


def test_protected_ids_query_reads_only_current_candidates(config):
    class Client:
        def query(self, sql, **kwargs):
            assert "annotations" not in sql and "candidates" in sql
            assert "DISTINCT" in sql
            assert "WHERE" not in sql and "JOIN" not in sql
            return Job([{"window_id": CLIP_A}, {"window_id": CLIP_B}])

    assert BigQueryStore(config, Client()).protected_window_ids() == {CLIP_A, CLIP_B}


def test_cleanup_aborts_if_protection_query_fails(config, tmp_path):
    client = StorageClient()
    storage = StorageStore(config, client)
    add_clip(storage, tmp_path, CLIP_A)

    class Denied(MetadataStore):
        def protected_window_ids(self):
            raise PermissionError("BigQuery denied")

    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        cleanup_cloud_audio(config, Catalog(tmp_path / "unused"), Denied(), storage)
    assert f"windows/v1/{CLIP_A}.wav" in client.blobs
    report = json.loads((config.artifacts / "cloud-cleanup.json").read_text())
    assert report["state"] == "cleanup_incomplete"
    assert "windows/v1/.publication.lock" not in client.blobs


def test_generation_preconditions_and_path_boundaries(config, tmp_path):
    client = StorageClient()
    storage = StorageStore(config, client)
    add_clip(storage, tmp_path, CLIP_A)
    blob = storage.list_window_audio()[CLIP_A]
    client.blobs[blob.name].generation += 1
    with pytest.raises(PreconditionFailed):
        storage.delete_window_audio(CLIP_A, blob)
    assert blob.name in client.blobs
    with pytest.raises(ValueError, match="canonical"):
        storage.delete_window_audio("not-a-uuid", blob)
    wrong = copy(blob)
    wrong.name = f"reference_catalog/{CLIP_A}.wav"
    with pytest.raises(ValueError, match="outside"):
        storage.delete_window_audio(CLIP_A, wrong)
    wrong = copy(blob)
    wrong.bucket = SimpleNamespace(name="different-bucket")
    with pytest.raises(ValueError, match="outside"):
        storage.delete_window_audio(CLIP_A, wrong)
    del client.blobs[blob.name]
    assert storage.delete_window_audio(CLIP_A, blob) is False


def test_shared_lock_excludes_other_clients_and_releases_after_failure(config):
    client = StorageClient()
    first = StorageStore(config, client)
    second = StorageStore(config, client)
    with pytest.raises(RuntimeError, match="simulated crash"):
        with first.publication_lock():
            with pytest.raises(RuntimeError, match="publication lock"):
                with second.publication_lock():
                    pytest.fail("Overlapping publication was allowed")
            raise RuntimeError("simulated crash")
    with second.publication_lock():
        assert "windows/v1/.publication.lock" in client.blobs
    assert "windows/v1/.publication.lock" not in client.blobs


def test_cleanup_retry_uses_new_queue_without_republishing(config, tmp_path, monkeypatch):
    audio = tmp_path / "w.wav"
    audio.write_bytes(b"window audio")
    catalog = Catalog(audio)
    bq = MetadataStore()
    bq.fail = False
    client = StorageClient()
    storage = StorageStore(config, client)
    path = write_candidates(tmp_path, [candidate()])
    publish_candidates(config, catalog, path, 1, "run", bq, storage)
    add_clip(storage, tmp_path, CLIP_A)
    name = f"windows/v1/{CLIP_A}.wav"
    client.delete_errors[name] = PermissionError("delete denied")
    manifest = tmp_path / "publish.json"
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        cleanup_cloud_audio(config, catalog, bq, storage, publication_manifest=manifest)
    assert json.loads(manifest.read_text())["state"] == "cleanup_incomplete"
    # A newer publication now needs the clip that previously failed deletion.
    bq.protected = {CLIP_A}
    client.delete_errors.clear()
    monkeypatch.setattr(
        bq, "replace_candidates", lambda *args: pytest.fail("Retry must not replace the queue")
    )
    # Turning cleanup off must not silently resume deletion during a publish retry.
    publish_candidates(config, catalog, path, 1, "run", bq, storage)
    assert json.loads(manifest.read_text())["state"] == "cleanup_incomplete"
    publish_candidates(config, catalog, path, 1, "run", bq, storage, cleanup=True)
    assert name in client.blobs
    assert json.loads(manifest.read_text())["state"] == "complete"


def test_current_candidate_can_be_repaired_from_local_audio(config, tmp_path, monkeypatch):
    client = StorageClient()
    storage = StorageStore(config, client)
    path = add_clip(storage, tmp_path, CLIP_A)
    catalog = Catalog(path)
    catalog.window = WindowRecord(CLIP_A, "r", 0, 5, 1)
    catalog.registered.add(CLIP_A)
    bq = MetadataStore()
    bq.fail = False
    cleanup_cloud_audio(config, catalog, bq, storage)
    assert f"windows/v1/{CLIP_A}.wav" not in client.blobs
    bq.protected.add(CLIP_A)  # New queue needs a previously removed clip.
    cleanup_cloud_audio(config, catalog, bq, storage)
    assert client.blobs[f"windows/v1/{CLIP_A}.wav"].data == path.read_bytes()
    assert catalog.registered == {CLIP_A}


def test_cleanup_retries_partial_deletion_without_a_per_clip_journal(config, tmp_path):
    client = StorageClient()
    storage = StorageStore(config, client)
    for window_id in [CLIP_A, CLIP_B]:
        add_clip(storage, tmp_path, window_id)
    name = f"windows/v1/{CLIP_B}.wav"
    client.delete_errors[name] = PermissionError("delete denied")
    bq = MetadataStore()
    catalog = Catalog(tmp_path / "unused")
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        cleanup_cloud_audio(config, catalog, bq, storage)
    assert f"windows/v1/{CLIP_A}.wav" not in client.blobs
    assert name in client.blobs
    client.delete_errors.clear()
    result = cleanup_cloud_audio(config, catalog, bq, storage)
    assert result["removed_clips"] == 1
    assert not client.blobs


def test_interrupt_retains_lock_while_remote_query_could_still_be_running(config):
    client = StorageClient()
    storage = StorageStore(config, client)
    with pytest.raises(KeyboardInterrupt):
        with storage.publication_lock():
            raise KeyboardInterrupt()
    assert "windows/v1/.publication.lock" in client.blobs


def test_cleanup_logs_storage_totals_after_deletion(config, tmp_path, caplog):
    client = StorageClient()
    storage = StorageStore(config, client)
    add_clip(storage, tmp_path, CLIP_A)
    add_clip(storage, tmp_path, CLIP_B)
    # Reference bytes count toward whole-bucket usage too.
    client.blob("reference_catalog/example.mp3").upload_from_string(
        "reference", if_generation_match=0
    )
    bq = MetadataStore()
    bq.protected = {CLIP_A}
    with caplog.at_level("INFO"):
        result = cleanup_cloud_audio(config, Catalog(tmp_path / "unused"), bq, storage)
    usage = result["storage_usage"]
    assert usage["gcs"]["soft_deleted_bytes"] == len(b"local window audio")
    assert usage["gcs"]["active_bytes"] >= len(b"local window audio") + len("reference")
    assert usage["bigquery"]["logical_bytes"] == 1000
    assert caplog.text.index("Cleanup complete:") < caplog.text.index("GCS storage:")
    assert "BigQuery storage: 0.000001 GB" in caplog.text
    saved = json.loads((config.artifacts / "cloud-cleanup.json").read_text())
    assert saved["storage_usage"] == usage


def test_bigquery_usage_reads_table_metadata_without_scanning_rows(config):
    class Client:
        def list_tables(self, dataset):
            assert dataset == "osaanurans.osafrogdataset"
            return [SimpleNamespace(reference=name) for name in ["windows", "annotations", "view"]]

        def get_table(self, reference):
            return SimpleNamespace(
                num_bytes={"windows": 100, "annotations": 200, "view": None}[reference]
            )

    assert BigQueryStore(config, Client()).storage_usage() == {"logical_bytes": 300}


def test_usage_reporting_failure_does_not_fail_completed_cleanup(
    config, tmp_path, monkeypatch, caplog
):
    client = StorageClient()
    storage = StorageStore(config, client)
    bq = MetadataStore()

    def denied():
        raise PermissionError("metadata access denied")

    monkeypatch.setattr(bq, "storage_usage", denied)
    result = cleanup_cloud_audio(config, Catalog(tmp_path / "unused"), bq, storage)
    assert result["state"] == "complete"
    assert "gcs" in result["storage_usage"]
    assert "bigquery" not in result["storage_usage"]
    assert "storage usage unavailable" in caplog.text


def test_publication_enriches_a_copy_without_changing_saved_selection(config, tmp_path):
    audio = tmp_path / "w.wav"
    audio.write_bytes(b"window audio")
    catalog = Catalog(audio)
    catalog.window = WindowRecord("w", "r", 30, 35, 1)
    bq = MetadataStore()
    bq.fail = False
    selection = write_candidates(tmp_path, [candidate()])
    original = selection.read_bytes()
    captured = []

    def replace_queue(path, count, run):
        validate_candidates(path, count, published=True)
        captured.extend(json.loads(line) for line in path.read_text().splitlines())

    bq.replace_candidates = replace_queue
    publish_candidates(
        config, catalog, selection, 1, "run", bq, StorageStore(config, StorageClient())
    )
    assert selection.read_bytes() == original
    assert captured == [
        dict(
            candidate(),
            recording_id="r",
            recording_start_seconds=30,
            recording_end_seconds=35,
            audio_gcs_uri="gs://osa-frog-data/windows/v1/w.wav",
            spectrogram_gcs_uri=None,
        )
    ]


def test_cleanup_deletes_annotation_only_audio_but_keeps_reference_inside_window_prefix(
    config, tmp_path
):
    client = StorageClient()
    storage = StorageStore(config, client)
    for key in [CLIP_A, CLIP_B, CLIP_C]:
        add_clip(storage, tmp_path, key)
    bq = MetadataStore()
    bq.protected = {CLIP_A}  # Current candidate; CLIP_C is annotation-only.
    bq.references = {f"gs://{config.bucket}/windows/v1/{CLIP_B}.wav"}
    result = cleanup_cloud_audio(config, Catalog(tmp_path / "unused"), bq, storage)
    assert result["removed_clips"] == 1
    assert set(client.blobs) == {f"windows/v1/{CLIP_A}.wav", f"windows/v1/{CLIP_B}.wav"}


def test_annotation_refresh_reads_only_answers(config):
    class Client:
        def query(self, sql, **kwargs):
            assert sql == "SELECT * FROM `osaanurans.osafrogdataset.annotations`"
            return Job([{"window_id": "w", "query_species_id": "A", "response": "present"}])

    assert BigQueryStore(config, Client()).fetch_annotations()[0]["window_id"] == "w"


@pytest.mark.parametrize(
    "changes",
    [
        {"recording_start_seconds": float("nan")},
        {"recording_end_seconds": 6},
        {"recording_id": ""},
        {"audio_gcs_uri": "gs://other/windows/v1/w.wav"},
        {"spectrogram_gcs_uri": "gs://other/image.png"},
    ],
)
def test_bad_enriched_candidate_never_loads_or_replaces(config, tmp_path, changes):
    row = dict(upload_candidate(), **changes)
    client = BigQueryClient()
    with pytest.raises(ValueError):
        BigQueryStore(config, client).replace_candidates(
            write_candidates(tmp_path, [row]), 1, "run"
        )
    assert client.loads == 0
    assert client.current[0]["candidate_id"] == "old"
