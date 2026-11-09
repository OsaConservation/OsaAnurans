"""Reference onboarding with real audio, tiny embeddings, and isolated cloud clients."""

import json
import shutil
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from google.cloud.bigquery import SchemaField
from osa_anurans_app.src import cli, console_ui, species
from osa_anurans_app.src.annotations import load_snapshot, save_snapshot
from osa_anurans_app.src.catalog import Catalog
from osa_anurans_app.src.cloud.species import TABLE_KEYS, insert_batch
from osa_anurans_app.src.cloud.storage import StorageStore
from osa_anurans_app.src.config import AppConfig
from osa_anurans_app.src.files import write_json
from osa_anurans_app.src.records import LabelCounts
from osa_anurans_app.tests.test_cloud import StorageClient
from osa_anurans_app.tests.test_pipeline import TinyBackend

FIELDS = {
    "species": "species_id scientific_name common_name",
    "reference_media": "reference_id species_id audio_gcs_uri spectrogram_gcs_uri",
    "recordings": "recording_id",
    "windows": "window_id recording_id start_seconds end_seconds",
    "annotations": "annotation_id window_id candidate_id query_species_id response annotator_id annotator_name submission_id created_at",
}


class SpeciesBQ:
    """Simulate atomic insertion and conflict failures; also inspect generated SQL."""

    dataset = "test.dataset"

    def __init__(self):
        self.tables = {table: [] for table in TABLE_KEYS}
        self.staging = {}
        self.commits = 0
        self.fail = False
        self.refresh_fail = False
        self.client = self
        self.deleted = []

    def get_table(self, name):
        table = name.rsplit(".", 1)[-1]
        fields = [
            SchemaField(
                field,
                "FLOAT64"
                if field.endswith("seconds")
                else "TIMESTAMP"
                if field == "created_at"
                else "STRING",
            )
            for field in FIELDS[table].split()
        ]
        return SimpleNamespace(schema=fields)

    def _table(self, name):
        return f"`{self.dataset}.{name}`"

    def _load_rows(self, rows, stage, schema):
        table = next(table for table in TABLE_KEYS if stage.startswith(f"_species_{table}_"))
        self.staging[table] = deepcopy(rows)

    def delete_table(self, destination, **kwargs):
        self.deleted.append(destination)

    def _query(self, sql):
        if sql.startswith("SELECT *"):
            table = sql.split(".")[-1].rstrip("`")
            return deepcopy(self.tables[table])
        assert sql.startswith("BEGIN TRANSACTION;")
        assert sql.rstrip().endswith("COMMIT TRANSACTION;")
        assert "TRUNCATE" not in sql
        assert "Scientific name already has another species ID" in sql
        assert sql.count("Duplicate existing") == 5
        assert sql.count("Conflicting existing") == 5
        assert "IS DISTINCT FROM" in sql
        if self.fail:
            raise RuntimeError("transaction failed")
        result = deepcopy(self.tables)
        for table, key in TABLE_KEYS.items():
            for row in self.staging[table]:
                found = [old for old in result[table] if old[key] == row[key]]
                if len(found) > 1 or any(
                    old.get(field) != value
                    for old in found
                    for field, value in row.items()
                    if field != "created_at"
                ):
                    raise ValueError("Conflicting existing metadata")
                if not found:
                    result[table].append(row)
        self.tables = result
        self.commits += 1
        return []

    def fetch_annotations(self):
        if self.refresh_fail:
            raise RuntimeError("refresh failed")
        return deepcopy(self.tables["annotations"])


class SpeciesStorage:
    """Create-only file objects, with a non-reentrant publication lock."""

    def __init__(self, config):
        self.config = config
        self.objects = {}
        self.locked = False
        self.writes = 0

    @contextmanager
    def publication_lock(self):
        assert not self.locked
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    def _upload(self, name, path):
        assert self.locked
        uri = f"gs://{self.config.bucket}/{name}"
        data = path.read_bytes()
        if uri in self.objects and self.objects[uri] != data:
            raise ValueError("Conflicting object")
        self.objects[uri] = data
        self.writes += 1
        return uri

    def upload_reference(self, species_id, reference_id, path):
        return self._upload(f"references/{species_id}/{reference_id}.wav", path)

    def upload_window(self, window, path):
        return self._upload(f"{self.config.prefix}/{window.window_id}.wav", path)

    def download_reference(self, uri, path):
        path.write_bytes(self.objects[uri])


@pytest.fixture
def setup(tmp_path):
    config = AppConfig((tmp_path / "missing-inbox",), None, tmp_path / "artifacts")
    audio = tmp_path / "reference.wav"
    sf.write(audio, np.full(160_000, 0.2, dtype=np.float32), 32_000, subtype="FLOAT")
    manifest = tmp_path / "species.json"
    manifest.write_text(
        json.dumps(
            {
                "species": [
                    {
                        "scientific_name": "Smilisca phaeota",
                        "common_name": "Test frog",
                        "reference_audio": audio.name,
                    }
                ]
            }
        ),
        encoding="utf-8-sig",
    )
    return config, manifest, audio, SpeciesBQ(), SpeciesStorage(config)


def run(setup, **kwargs):
    config, manifest, _, bq, storage = setup
    return species.add_species(
        config,
        species.load_manifest(manifest),
        backend=TinyBackend(),
        bq=bq,
        storage=storage,
        **kwargs,
    )


def test_add_and_retry_preserve_votes_timestamps_and_complete_snapshot(setup):
    config, _, _, bq, storage = setup
    bq.tables["annotations"].append(
        {
            "window_id": "older",
            "query_species_id": "OTHER",
            "response": "absent",
            "annotation_id": "old",
        }
    )
    first = run(setup)
    original = deepcopy(bq.tables)
    assert len(load_snapshot(first)) == 2
    assert len(bq.tables["annotations"]) == 2
    assert bq.tables["annotations"][-1]["annotator_id"] == "new-species"
    assert bq.tables["reference_media"][0]["spectrogram_gcs_uri"] is None
    # Correctness must not depend on finding the previous attempt receipt.
    shutil.rmtree(config.artifacts / "species-imports")
    second = run(setup)
    assert second != first
    assert bq.tables == original
    assert len(storage.objects) == 1
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        assert len(catalog.embedding_rows(TinyBackend.config_id)) == 1
        assert not catalog.pending_windows()


@pytest.mark.parametrize("subtype,offset", [("PCM_16", 0), ("FLOAT", 5e-7)])
def test_equivalent_prepared_references_are_rejected_before_cloud_writes(setup, subtype, offset):
    _, manifest, audio, bq, storage = setup
    samples = np.full(160_000, 0.25, dtype=np.float32)
    sf.write(audio, samples, 32_000, subtype="FLOAT")
    second = audio.with_name("equivalent.wav")
    sf.write(second, samples + offset, 32_000, subtype=subtype)
    assert audio.read_bytes() != second.read_bytes()
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    data["species"].append({**data["species"][0], "reference_audio": second.name})
    manifest.write_text(json.dumps(data))
    assert len(species.load_manifest(manifest).entries) == 2

    # Both the initial attempt and retry must fail before creating cloud data.
    for _ in range(2):
        with pytest.raises(
            ValueError, match="Duplicate prepared reference audio for SMIPHA"
        ) as error:
            run(setup)
        assert audio.name in str(error.value)
        assert second.name in str(error.value)
        assert bq.commits == storage.writes == 0
        assert all(not rows for rows in bq.tables.values())
        assert storage.objects == {}


def test_matching_legacy_reference_reused_with_spectrogram(setup):
    _, _, audio, bq, storage = setup
    bq.tables["species"] = [
        {"species_id": "SMIPHA", "scientific_name": "Smilisca phaeota", "common_name": "Test frog"}
    ]
    ref = {
        "reference_id": "SMIPHA_1",
        "species_id": "SMIPHA",
        "audio_gcs_uri": "gs://osa-frog-data/reference_catalog/SMIPHA_1.wav",
        "spectrogram_gcs_uri": "gs://osa-frog-data/reference_catalog/SMIPHA_1.webp",
    }
    bq.tables["reference_media"] = [ref.copy()]
    storage.objects[ref["audio_gcs_uri"]] = audio.read_bytes()
    run(setup)
    assert bq.tables["reference_media"] == [ref]
    assert len(bq.tables["annotations"]) == 1
    assert not any("/references/" in uri for uri in storage.objects)


@pytest.mark.parametrize("change", ["different-name", "different-id", "duplicate"])
def test_catalog_collisions_cause_no_audio_uploads(setup, change):
    _, _, _, bq, storage = setup
    row = {
        "species_id": "SMIPHA",
        "scientific_name": "Smilisca phaeota",
        "common_name": "Test frog",
    }
    if change == "different-name":
        row["common_name"] = "Other name"
    elif change == "different-id":
        row["species_id"] = "WRONG"
    bq.tables["species"] = [row] * (2 if change == "duplicate" else 1)
    with pytest.raises(ValueError):
        run(setup)
    assert storage.writes == bq.commits == 0


@pytest.mark.parametrize("stage", ["transaction", "refresh", "registration"])
def test_failure_retry_is_add_only(setup, monkeypatch, stage):
    config, _, _, bq, storage = setup
    old_snapshot = save_snapshot(config.artifacts / "old", [], "bigquery")
    write_json(
        config.artifacts / "latest.json",
        {"config_id": TinyBackend.config_id, "snapshot": str(old_snapshot)},
    )
    if stage == "transaction":
        bq.fail = True
    elif stage == "refresh":
        bq.refresh_fail = True
    else:
        original = Catalog.mark_registered
        monkeypatch.setattr(
            Catalog,
            "mark_registered",
            lambda *args: (_ for _ in ()).throw(RuntimeError("registration failed")),
        )
    with pytest.raises(RuntimeError):
        run(setup)
    assert json.loads((config.artifacts / "latest.json").read_text())["snapshot"] == str(
        old_snapshot
    )
    assert len(bq.tables["annotations"]) == (0 if stage == "transaction" else 1)
    assert len(storage.objects) == 1
    if stage == "registration":
        monkeypatch.setattr(Catalog, "mark_registered", original)
    bq.fail = bq.refresh_fail = False
    run(setup)
    assert len(bq.tables["annotations"]) == 1


def test_incompatible_config_and_nonempty_csv_snapshot_block_cloud_writes(setup):
    config, _, _, bq, storage = setup
    write_json(config.artifacts / "latest.json", {"config_id": "other"})
    with pytest.raises(ValueError, match="configuration differs"):
        run(setup)
    snapshot = save_snapshot(config.artifacts / "csv", [LabelCounts("w", "s", 1, 0)], "initial.csv")
    write_json(config.artifacts / "latest.json", {"snapshot": str(snapshot)})
    with pytest.raises(ValueError, match="CSV seed workflow"):
        run(setup)
    assert bq.commits == storage.writes == 0


@pytest.mark.parametrize("edit", ["manifest", "audio"])
def test_changed_reviewed_inputs_are_rejected(setup, edit):
    config, manifest_path, audio, bq, storage = setup
    manifest = species.load_manifest(manifest_path)
    path = manifest_path if edit == "manifest" else audio
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after review"):
        species.add_species(config, manifest, backend=TinyBackend(), bq=bq, storage=storage)
    assert not config.artifacts.exists()


@pytest.mark.parametrize("kind", ["short", "long", "silence", "cancel", "nan", "corrupt"])
def test_invalid_audio_rejected_before_cloud(setup, kind):
    _, manifest, audio, bq, storage = setup
    length = 159_999 if kind == "short" else 160_001 if kind == "long" else 160_000
    samples = np.full(length, 0.2, dtype=np.float32)
    if kind == "silence":
        samples[:] = 0
    if kind == "nan":
        samples[0] = np.nan
    if kind == "cancel":
        samples = np.stack([samples, -samples], axis=1)
    sf.write(audio, samples, 32_000, subtype="FLOAT")
    if kind == "corrupt":
        audio.write_bytes(b"broken")
    with pytest.raises(ValueError, match="entry 1"):
        species.load_manifest(manifest)
    assert bq.commits == storage.writes == 0


@pytest.mark.parametrize(
    "data", [None, {}, {"species": []}, {"species": [{}]}, {"species": [{"scientific_name": 3}]}]
)
def test_malformed_manifest(setup, data):
    _, manifest, *_ = setup
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        species.load_manifest(manifest)


def test_name_normalization_duplicates_and_multiple_references(setup):
    config, path, audio, bq, storage = setup
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    data["species"][0]["scientific_name"] = "  smilisca   PHAEOTA "
    path.write_text(json.dumps(data))
    assert species.load_manifest(path).entries[0].species_id == "SMIPHA"
    data["species"].append(data["species"][0].copy())
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate species/reference"):
        species.load_manifest(path)
    second = audio.with_name("second.wav")
    sf.write(second, np.full(160_000, 0.4, dtype=np.float32), 32_000, subtype="FLOAT")
    data["species"][1]["reference_audio"] = second.name
    path.write_text(json.dumps(data))
    run(setup)
    assert len(bq.tables["species"]) == 1
    assert len(bq.tables["reference_media"]) == len(bq.tables["annotations"]) == 2


def test_reference_upload_is_create_only_and_outside_cleanup_prefix(tmp_path):
    config = AppConfig((tmp_path,), None, tmp_path)
    client = StorageClient()
    storage = StorageStore(config, client)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"waveform")
    reference_id = "433f8993-20bc-4fd5-9969-0c8fe22a5840"
    uri = storage.upload_reference("SMIPHA", reference_id, audio)
    assert uri.startswith("gs://osa-frog-data/references/SMIPHA/")
    assert storage.upload_reference("SMIPHA", reference_id, audio) == uri
    assert storage.list_window_audio() == {}
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        storage.upload_reference("SMIPHA", reference_id, audio)


def test_transaction_conflict_preserves_entire_batch(setup):
    _, _, _, bq, _ = setup
    run(setup)
    before = deepcopy(bq.tables)
    changed = deepcopy(before)
    changed["annotations"][0]["response"] = "absent"
    with pytest.raises(ValueError, match="Conflicting"):
        insert_batch(bq, changed)
    assert bq.tables == before


def test_workflow_preserves_review_payload_and_reuses_refresh(setup, monkeypatch):
    config, path, *_ = setup
    manifest = species.load_manifest(path)
    workflow = console_ui.Workflow.from_selection(["train", "refresh", "add-species"], manifest)
    assert workflow.steps == ("add-species", "refresh", "train")
    calls = []
    monkeypatch.setattr(species, "add_species", lambda c, m: calls.append(m))
    monkeypatch.setattr(cli, "refresh_annotations", lambda *args: pytest.fail("Redundant refresh"))
    monkeypatch.setattr(cli, "train_local", lambda *args: calls.append("train"))
    cli.execute_workflow(workflow, config)
    assert calls == [manifest, "train"]
    assert cli.parser().parse_args(["add-species", "--manifest", str(path)]).manifest == path
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["add-species"])


def test_failed_add_stops_training_and_publication(setup, monkeypatch):
    config, path, *_ = setup

    def fail(*args):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(species, "add_species", fail)
    monkeypatch.setattr(cli, "train_local", lambda *args: pytest.fail("Unexpected training"))
    workflow = console_ui.Workflow.from_selection(
        ["add-species", "publish"], species.load_manifest(path)
    )
    with pytest.raises(RuntimeError):
        cli.execute_workflow(workflow, config)


def test_relocated_input_keeps_seed_identity(setup):
    config, path, audio, bq, storage = setup
    run(setup)
    original = deepcopy(bq.tables)
    moved = path.parent / "moved"
    moved.mkdir()
    shutil.copy(audio, moved / audio.name)
    shutil.copy(path, moved / path.name)
    species.add_species(
        config,
        species.load_manifest(moved / path.name),
        backend=TinyBackend(),
        bq=bq,
        storage=storage,
    )
    assert bq.tables == original


def test_supplied_mp3_templates_decode_as_five_seconds():
    root = Path("C:/Users/DELL EMC/Desktop/Grillos/Templates")
    if not root.exists():
        pytest.skip("User template directory not available")
    paths = list(root.glob("*.mp3"))
    assert len(paths) == 12
    for path in paths:
        species.validate_audio(path)


def test_add_then_train_uses_corpus_and_excludes_seed(setup):
    from osa_anurans_app.src.pipeline import import_local, train_local

    config, _, audio, bq, _ = setup
    inbox = config.audio_roots[0]
    inbox.mkdir()
    sf.write(inbox / "corpus.wav", np.full(480_000, 0.3, dtype=np.float32), 32_000, subtype="FLOAT")
    import_local(config, backend=TinyBackend())
    run(setup)
    directory = train_local(config)
    rows = [json.loads(line) for line in (directory / "candidates.jsonl").read_text().splitlines()]
    seed_id = bq.tables["annotations"][0]["window_id"]
    assert len(rows) == 3
    assert all(
        row["query_species_id"] == "SMIPHA" and row["selection_method"] == "similarity"
        for row in rows
    )
    assert all(row["window_id"] != seed_id for row in rows)


def test_mono_conversion_failure_after_backend_blocks_cloud(setup):
    config, path, _, bq, storage = setup

    class SilentBackend(TinyBackend):
        def prepare_audio(self, path):
            return np.zeros((1, 160_000), dtype=np.float32)

    with pytest.raises(ValueError, match="nonzero model frame"):
        species.add_species(
            config, species.load_manifest(path), backend=SilentBackend(), bq=bq, storage=storage
        )
    assert bq.commits == storage.writes == 0


def test_multiple_species_share_one_window_and_keep_both_seed_votes(setup):
    _, path, _, bq, _ = setup
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    data["species"].append(
        {**data["species"][0], "scientific_name": "Boana rosenbergi", "common_name": "Other frog"}
    )
    path.write_text(json.dumps(data))
    snapshot = run(setup)
    assert len(bq.tables["windows"]) == len(bq.tables["recordings"]) == 1
    assert len(load_snapshot(snapshot)) == len(bq.tables["reference_media"]) == 2


def test_menu_reads_configured_manifest_without_path_prompt(setup, monkeypatch, capsys):
    config, path, *_ = setup
    config = replace(config, species_manifest=path)
    selections = iter([["add-species"], ["add-species", "train"]])
    decisions = iter(["Go back", "Start workflow"])
    monkeypatch.setattr(console_ui, "_workflow_menu", lambda *args: next(selections))
    monkeypatch.setattr(
        console_ui.questionary,
        "path",
        lambda *args, **kwargs: pytest.fail("Unexpected path prompt"),
    )
    monkeypatch.setattr(console_ui, "_select", lambda *args: next(decisions))
    workflow = console_ui.configure_workflow(config)
    assert workflow.species_manifest.path == path
    assert workflow.steps == ("add-species", "train")
    output = capsys.readouterr().out
    assert f"Read species from: {path}" in output
    assert "SMIPHA" in output
    assert "Upload 1 reference clip and add presence labels." in output
    assert "gs://" not in output
    assert config.dataset not in output
    assert not config.artifacts.exists()


def test_missing_configured_manifest_returns_to_menu_without_writes(setup, monkeypatch, capsys):
    config, path, *_ = setup
    config = replace(config, species_manifest=path.with_name("missing.json"))
    selections = iter([["add-species"], "exit"])
    monkeypatch.setattr(console_ui, "_workflow_menu", lambda *args: next(selections))
    monkeypatch.setattr(console_ui, "_select", lambda *args: pytest.fail("Unexpected confirmation"))
    assert console_ui.configure_workflow(config) == "exit"
    assert str(config.species_manifest_path) in capsys.readouterr().out
    assert not config.artifacts.exists()


def test_prior_import_csv_snapshot_is_rechecked_when_add_executes(setup, monkeypatch):
    config, path, *_ = setup
    manifest = species.load_manifest(path)

    def import_csv(config):
        snapshot = save_snapshot(
            config.artifacts / "csv", [LabelCounts("w", "s", 1, 0)], "import.csv"
        )
        write_json(config.artifacts / "latest.json", {"snapshot": str(snapshot)})

    monkeypatch.setattr(cli, "import_local", import_csv)
    workflow = console_ui.Workflow.from_selection(["import", "add-species", "train"], manifest)
    with pytest.raises(ValueError, match="CSV seed workflow"):
        cli.execute_workflow(workflow, config)


def test_cleanup_removes_seed_window_audio_but_retains_reference(setup):
    from osa_anurans_app.src.publish import cleanup_cloud_audio

    config, _, _, bq, storage = setup
    run(setup)
    storage_client = StorageClient()
    actual = StorageStore(config, storage_client)
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        window = catalog.windows()[0]
        actual.upload_window(window, catalog.audio_path(window.window_id))
        reference = bq.tables["reference_media"][0]
        actual.upload_reference(
            "SMIPHA", reference["reference_id"], catalog.audio_path(window.window_id)
        )
        bq.protected_window_ids = lambda: set()
        bq.reference_audio_uris = lambda: {reference["audio_gcs_uri"]}
        bq.storage_usage = lambda: {"logical_bytes": 0}
        cleanup_cloud_audio(config, catalog, bq, actual)
        assert len([blob for blob in storage_client.blobs.values() if blob.data]) == 1


def test_upload_failure_leaves_metadata_untouched_and_retry_reuses_audio(setup, monkeypatch):
    _, _, _, bq, storage = setup
    original = storage.upload_reference

    def fail(*args):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(storage, "upload_reference", fail)
    with pytest.raises(RuntimeError, match="upload failed"):
        run(setup)
    assert all(not rows for rows in bq.tables.values())
    assert len(storage.objects) == 0
    monkeypatch.setattr(storage, "upload_reference", original)
    run(setup)
    assert len(bq.tables["annotations"]) == 1
    assert len(storage.objects) == 1


@pytest.mark.parametrize("supplied", ["bq", "storage"])
def test_partial_cloud_client_injection_is_preserved(setup, monkeypatch, supplied):
    config, path, _, expected_bq, expected_storage = setup

    def connect(config_arg, bq=None, storage=None):
        assert config_arg is config
        assert bq is (expected_bq if supplied == "bq" else None)
        assert storage is (expected_storage if supplied == "storage" else None)
        return expected_bq, expected_storage

    monkeypatch.setattr("osa_anurans_app.src.publish.check_cloud_access", connect)
    species.add_species(
        config,
        species.load_manifest(path),
        backend=TinyBackend(),
        **{supplied: expected_bq if supplied == "bq" else expected_storage},
    )
    assert expected_bq.commits == 1


def test_frozen_audio_is_revalidated_before_padding_can_hide_an_edit(setup, monkeypatch):
    config, path, audio, bq, storage = setup
    original = species.validate_audio
    calls = 0

    def validate_then_edit(path):
        nonlocal calls
        original(path)
        calls += 1
        if calls == 1:
            # Simulate an editor changing audio between validation and hashing.
            sf.write(audio, np.full(128_000, 0.2, dtype=np.float32), 32_000, subtype="FLOAT")

    monkeypatch.setattr(species, "validate_audio", validate_then_edit)
    manifest = species.load_manifest(path)
    with pytest.raises(ValueError, match="exactly five decoded seconds"):
        species.add_species(config, manifest, backend=TinyBackend(), bq=bq, storage=storage)
    assert bq.commits == storage.writes == 0
