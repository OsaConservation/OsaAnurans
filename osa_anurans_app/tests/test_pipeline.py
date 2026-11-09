"""Exercise the local workflow with real WAVs and a deterministic tiny backend."""

import builtins
import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from osa_anurans_app.src.annotations import load_snapshot, save_snapshot
from osa_anurans_app.src.audio import SAMPLE_RATE, WINDOW_SAMPLES
from osa_anurans_app.src.catalog import Catalog
from osa_anurans_app.src.config import AppConfig, load_config
from osa_anurans_app.src.model.model import load_artifact
from osa_anurans_app.src.pipeline import import_local, train_local


class TinyBackend:
    """No neural runtime or cloud clients; infer directly from waveform means."""

    config_id = "test-perch-shaped-backend"
    metadata = {"model": "test-only", "window_samples": WINDOW_SAMPLES}

    def __init__(self, fail_if_called=False):
        self.fail_if_called = fail_if_called
        self.prepared = {}
        self.embed_calls = 0

    def prepare_audio(self, path):
        if self.fail_if_called:
            raise AssertionError("A completed recording should not be prepared again")
        audio, sample_rate = sf.read(path, dtype="float32")
        assert sample_rate == SAMPLE_RATE
        padding = (-len(audio)) % WINDOW_SAMPLES
        frames = np.pad(audio, (0, padding), mode="wrap").reshape(-1, WINDOW_SAMPLES)
        self.prepared[path.resolve()] = frames.copy()
        return frames

    def embed_frames(self, frames):
        if self.fail_if_called:
            raise AssertionError("A completed recording should not invoke inference again")
        self.embed_calls += 1
        means = frames.mean(axis=1)
        return np.stack([means, -means], axis=1).astype(np.float32)


@pytest.fixture
def local_config(tmp_path):
    root = tmp_path / "Grillos"
    root.mkdir()
    annotations = root / "annotations.csv"
    rows = []
    for recording in range(5):
        # Distinct bytes retain five recording identities. Every recording has
        # both classes, plus an unknown short tail requiring wrap padding.
        amplitude = np.float32(0.2 + recording * 0.02)
        audio = np.concatenate(
            [
                np.full(WINDOW_SAMPLES, amplitude, dtype=np.float32),
                np.full(WINDOW_SAMPLES, -amplitude, dtype=np.float32),
                np.linspace(-0.1, 0.1, SAMPLE_RATE * 2, dtype=np.float32),
            ]
        )
        relative = f"deployment/recording-{recording}.wav"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
        rows.extend(
            [
                {"audiofilename": relative, "start": 0, "end": 5, "label:species": "A"},
                {"audiofilename": relative, "start": 5, "end": 10, "label:species": "B"},
            ]
        )
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["audiofilename", "start", "end", "label:species"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return AppConfig(
        (root,),
        annotations,
        tmp_path / "artifacts",
        device="cpu",
        epochs=30,
        learning_rate=0.1,
        batch_size=4,
    )


def prohibit_cloud_imports(monkeypatch):
    original_import = builtins.__import__

    def local_import(name, *args, **kwargs):
        if name == "google" or name.startswith("google."):
            raise AssertionError(f"Local workflow attempted a cloud import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", local_import)


def test_audio_only_import_preserves_labels_and_adds_unknown_windows(local_config):
    snapshot = import_local(local_config, backend=TinyBackend())
    inbox = local_config.artifacts.parent / "inbox"
    inbox.mkdir()
    sf.write(inbox / "new.wav", np.full(WINDOW_SAMPLES, 0.45), SAMPLE_RATE, subtype="FLOAT")
    config = replace(local_config, audio_roots=(inbox,), annotations_csv=None)
    assert import_local(config, backend=TinyBackend(), limit_recordings=1) == snapshot
    assert len(load_snapshot(snapshot)) == 10
    latest = json.loads((config.artifacts / "latest.json").read_text())
    assert latest["snapshot"] == str(snapshot)
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        assert len(catalog.windows()) == 16
        assert len(catalog.embedding_rows(TinyBackend.config_id)) == 16


def test_first_audio_only_import_does_not_create_presence_labels(local_config):
    config = replace(local_config, annotations_csv=None)
    assert import_local(config, backend=TinyBackend(), limit_recordings=1) is None
    latest = json.loads((config.artifacts / "latest.json").read_text())
    assert latest == {"config_id": TinyBackend.config_id}


def test_local_import_preserves_audio_embedding_alignment_and_reuses_cache(
    local_config, monkeypatch
):
    prohibit_cloud_imports(monkeypatch)
    backend = TinyBackend()
    snapshot = import_local(local_config, backend=backend)
    assert backend.embed_calls == 5
    assert len(load_snapshot(snapshot)) == 10

    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        windows = catalog.windows()
        rows = catalog.embedding_rows(backend.config_id)
        recordings = {recording.recording_id: recording for recording in catalog.recordings()}
        assert len(windows) == len(rows) == 15
        assert len(catalog.pending_windows()) == 15  # Local import must not upload anything.
        assert all(not hasattr(window, "audio_gcs_uri") for window in windows)
        window_by_id = {window.window_id: window for window in windows}
        identities = {window.window_id for window in windows}

        for row in rows:
            window = window_by_id[row.window_id]
            recording = recordings[row.recording_id]
            expected_frame = backend.prepared[recording.path][row.row_index]
            audio_path = catalog.audio_path(window.window_id)
            actual, sample_rate = sf.read(audio_path, dtype="float32")
            assert sample_rate == SAMPLE_RATE
            assert len(actual) == WINDOW_SAMPLES
            assert np.array_equal(actual, expected_frame)
            embeddings = np.load(row.path, allow_pickle=False)
            assert np.array_equal(
                embeddings[row.row_index],
                np.array(
                    [
                        expected_frame.mean(),
                        -expected_frame.mean(),
                    ],
                    dtype=np.float32,
                ),
            )
            assert window.start_sample == row.row_index * WINDOW_SAMPLES
            assert window.end_sample - window.start_sample == WINDOW_SAMPLES

    # A new snapshot is written, but no waveform preparation/model invocation
    # occurs, and IDs and consensus remain identical.
    cached = TinyBackend(fail_if_called=True)
    second_snapshot = import_local(local_config, backend=cached)
    assert second_snapshot != snapshot
    assert load_snapshot(second_snapshot) == load_snapshot(snapshot)
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        assert {window.window_id for window in catalog.windows()} == identities
        assert len(catalog.embedding_rows(backend.config_id)) == 15
    latest = json.loads((local_config.artifacts / "latest.json").read_text())
    assert latest["snapshot"] == str(second_snapshot)


def test_local_training_reloads_probe_and_covers_every_window_species(local_config, monkeypatch):
    prohibit_cloud_imports(monkeypatch)
    import_local(local_config, backend=TinyBackend())
    run_dir = train_local(local_config)
    artifact = load_artifact(run_dir)
    assert artifact.species_ids == ("A", "B")
    probabilities = artifact.predict_probabilities(
        np.array([[0.2, -0.2], [-0.2, 0.2]], dtype=np.float32)
    )
    assert probabilities.argmax(axis=1).tolist() == [0, 1]
    assert np.allclose(probabilities.sum(axis=1), 1)

    metadata = json.loads((run_dir / "candidates.json").read_text())
    candidates = [
        json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines()
    ]
    assert metadata["local_only"] is True
    assert metadata["count"] == len(candidates) == 24
    assert len({row["candidate_id"] for row in candidates}) == len(candidates)
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        window_ids = {window.window_id for window in catalog.windows()}
    assert all(row["window_id"] in window_ids for row in candidates)
    assert all(0 <= row["score"] <= 1 for row in candidates)
    assert not (run_dir / "publish.json").exists()

    with (run_dir / "split.csv").open(newline="", encoding="utf-8") as handle:
        split = list(csv.DictReader(handle))
    assert len(split) == 10  # Unknown tails never became absent/class labels.
    train_recordings = {row["recording_id"] for row in split if row["partition"] == "train"}
    test_recordings = {row["recording_id"] for row in split if row["partition"] == "test"}
    assert len(train_recordings) == 4
    assert len(test_recordings) == 1
    assert train_recordings.isdisjoint(test_recordings)
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        held_out_windows = {
            w.window_id for w in catalog.windows() if w.recording_id in test_recordings
        }
    assert not held_out_windows.intersection(row["window_id"] for row in candidates)
    assert json.loads((run_dir / "metrics.json").read_text())["test"]["accuracy"] == 1


@pytest.mark.parametrize("annotator_id", ["new-species", "eoin-template-seed"])
def test_training_excludes_reference_pairs_from_snapshot_provenance(local_config, annotator_id):
    import_local(local_config, backend=TinyBackend())
    latest = json.loads((local_config.artifacts / "latest.json").read_text())
    from pathlib import Path

    labels = load_snapshot(Path(latest["snapshot"]))
    references = {
        (label.window_id, label.species_id) for label in labels if label.species_id == "A"
    }
    answers = [
        {
            "window_id": window,
            "query_species_id": species,
            "annotator_id": annotator_id,
            "response": "present",
        }
        for window, species in references
    ]
    snapshot = save_snapshot(
        local_config.artifacts / "references", labels, "bigquery", raw_answers=answers
    )
    run = train_local(local_config, snapshot=snapshot)
    candidates = [json.loads(line) for line in (run / "candidates.jsonl").read_text().splitlines()]
    actual = {(row["window_id"], row["query_species_id"]) for row in candidates}
    assert not actual.intersection(references)
    ordinary = {(label.window_id, label.species_id) for label in labels if label.species_id == "B"}
    assert actual.intersection(ordinary)
    assert (
        json.loads((run / "candidate-selection.json").read_text())["species"]["A"][
            "positive_windows"
        ]
        > 0
    )


@pytest.mark.parametrize("explicit_manifest", [False, True])
def test_loaded_config_trains_and_saves_manifest_path(local_config, explicit_manifest):
    config_path = local_config.artifacts.parent / "loaded.toml"
    manifest_setting = "species_manifest = 'inputs/custom.json'\n" if explicit_manifest else ""
    config_path.write_text(
        "[paths]\n"
        f"audio_roots = {json.dumps([path.as_posix() for path in local_config.audio_roots])}\n"
        f"annotations_csv = {json.dumps(local_config.annotations_csv.as_posix())}\n"
        f"artifacts = {json.dumps(local_config.artifacts.as_posix())}\n"
        f"{manifest_setting}"
        "[training]\nepochs = 1\nbatch_size = 4\ndevice = 'cpu'\n"
    )
    config = load_config(config_path)
    import_local(config, backend=TinyBackend())
    directory = train_local(config)
    saved = json.loads((directory / "config.json").read_text())
    assert saved["species_manifest"] == str(config.species_manifest)
    assert saved["artifacts"] == str(config.artifacts)
    assert load_artifact(directory).species_ids == ("A", "B")
    assert json.loads((directory / "candidates.json").read_text())["count"] > 0


def test_failed_alignment_does_not_publish_partial_catalog_or_latest_snapshot(
    local_config, monkeypatch
):
    prohibit_cloud_imports(monkeypatch)

    class MisalignedBackend(TinyBackend):
        def embed_frames(self, frames):
            return super().embed_frames(frames)[:-1]

    with pytest.raises(ValueError, match="Embedding/window alignment mismatch"):
        import_local(local_config, backend=MisalignedBackend())
    assert not (local_config.artifacts / "latest.json").exists()
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        assert catalog.windows() == []
        assert catalog.embedding_rows(TinyBackend.config_id) == []
    reports = list((local_config.artifacts / "imports").glob("*/import.json"))
    report = json.loads(reports[0].read_text())
    assert "alignment mismatch" in report["failures"][0]["error"]

    backend = TinyBackend()
    snapshot = import_local(local_config, backend=backend)
    assert backend.embed_calls == 5
    assert len(load_snapshot(snapshot)) == 10


def test_training_uses_configured_candidate_limit(local_config):
    config = replace(local_config, candidates_per_species=2)
    import_local(config, backend=TinyBackend())
    run_dir = train_local(config)
    metadata = json.loads((run_dir / "candidates.json").read_text())
    candidates = [
        json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines()
    ]
    assert metadata["candidates_per_species"] == 2
    assert metadata["window_count"] == 15  # Full local pool remains available.
    assert metadata["count"] == len(candidates) == 4
    for species in ("A", "B"):
        assert sum(row["query_species_id"] == species for row in candidates) == 2


def test_missing_local_window_audio_triggers_only_affected_recording_rebuild(
    local_config, monkeypatch
):
    prohibit_cloud_imports(monkeypatch)
    import_local(local_config, backend=TinyBackend())
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        missing = catalog.audio_path(catalog.windows()[0].window_id)
    missing.unlink()

    backend = TinyBackend()
    import_local(local_config, backend=backend)
    assert backend.embed_calls == 1
    assert missing.is_file()


@pytest.mark.parametrize(
    "corruption",
    ["empty-wav", "short-wav", "empty-embedding", "row-index", "row-path", "row-swap"],
)
def test_invalid_cached_artifact_rebuilds_only_affected_recording(
    local_config, monkeypatch, corruption
):
    prohibit_cloud_imports(monkeypatch)
    import_local(local_config, backend=TinyBackend())
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        row = catalog.embedding_rows(TinyBackend.config_id)[0]
        audio = catalog.audio_path(row.window_id)
        if corruption in {"row-index", "row-path"}:
            other = catalog.embedding_rows(TinyBackend.config_id)[1]
            column, value = (
                ("row_index", 99)
                if corruption == "row-index"
                else ("path", str(local_config.artifacts / "missing.npy"))
            )
            with catalog.connection:
                catalog.connection.execute(
                    f"UPDATE window_embeddings SET {column}=? WHERE window_id=?",
                    (value, other.window_id),
                )
        elif corruption == "row-swap":
            other = catalog.embedding_rows(TinyBackend.config_id)[1]
            with catalog.connection:
                for window_id, index in [
                    (row.window_id, 99),
                    (other.window_id, 0),
                    (row.window_id, 1),
                ]:
                    catalog.connection.execute(
                        "UPDATE window_embeddings SET row_index=? WHERE window_id=?",
                        (index, window_id),
                    )
    if corruption == "empty-wav":
        audio.write_bytes(b"")
    elif corruption == "short-wav":
        sf.write(audio, np.zeros(100), SAMPLE_RATE, subtype="FLOAT")
    elif corruption == "empty-embedding":
        count = len(np.load(row.path))
        np.save(row.path, np.empty((count, 0), dtype=np.float32))
    backend = TinyBackend()
    import_local(local_config, backend=backend)
    assert backend.embed_calls == 1
    assert sf.info(audio).frames == WINDOW_SAMPLES
    assert np.load(row.path).shape[1] == 2
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        repaired = [
            item
            for item in catalog.embedding_rows(TinyBackend.config_id)
            if item.recording_id == row.recording_id
        ]
        assert [item.row_index for item in repaired] == [0, 1, 2]
        assert {item.path for item in repaired} == {row.path}


def test_import_accepts_whitespace_around_annotation_filenames(local_config):
    path = local_config.annotations_csv
    rows = list(csv.DictReader(path.read_text().splitlines()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "audiofilename": f" {row['audiofilename']} "} for row in rows)
    snapshot = import_local(local_config, backend=TinyBackend())
    assert len(load_snapshot(snapshot)) == 10


@pytest.mark.parametrize("multi_positive", [False, True])
def test_similarity_only_run_without_trainable_classifier(local_config, multi_positive):
    from osa_anurans_app.src.annotations import save_snapshot
    from osa_anurans_app.src.records import LabelCounts

    import_local(local_config, backend=TinyBackend())
    with Catalog(local_config.artifacts / "catalog.sqlite") as catalog:
        row = catalog.embedding_rows(TinyBackend.config_id)[0]
    labels = [LabelCounts(row.window_id, "ONLY", 1)]
    if multi_positive:
        labels.append(LabelCounts(row.window_id, "SECOND", 1))
    snapshot = save_snapshot(local_config.artifacts / "similarity-labels", labels, "test")
    run = train_local(local_config, snapshot=snapshot)
    metadata = json.loads((run / "candidates.json").read_text())
    assert metadata["classifier_available"] is False
    assert metadata["count"] > 0
    assert not (run / "model.pt").exists()
    assert not (run / "publish.json").exists()
