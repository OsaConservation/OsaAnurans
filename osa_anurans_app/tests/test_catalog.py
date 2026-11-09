import sqlite3
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from osa_anurans_app.src.catalog import Catalog
from osa_anurans_app.src.records import RecordingRecord, WindowRecord
from osa_anurans_app.src.sources import read_interval_labels


@pytest.fixture
def artifact(tmp_path):
    recording = RecordingRecord("r", tmp_path / "source.wav", "hash", 10, 100)
    windows = [WindowRecord("w0", "r", 0, 50, 10), WindowRecord("w1", "r", 50, 100, 10)]
    array_path = tmp_path / "embeddings.npy"
    np.save(array_path, np.arange(6, dtype=np.float32).reshape(2, 3))
    audio_paths = [tmp_path / f"{window.window_id}.wav" for window in windows]
    for path in audio_paths:
        sf.write(path, np.zeros(50), 10)
    return recording, windows, array_path, audio_paths


def test_artifact_registration_and_bidirectional_mapping(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        catalog.register_artifact(windows, "config", array_path, audio_paths)
        assert catalog.windows() == windows
        rows = catalog.embedding_rows("config")
        assert [(row.window_id, row.row_index) for row in rows] == [
            ("w0", 0),
            ("w1", 1),
        ]
        assert catalog.audio_path("w0") == audio_paths[0]
        catalog.register_artifact(windows, "second-config", array_path, audio_paths)
        assert len(catalog.windows()) == 2
        assert len(catalog.embedding_rows("second-config")) == 2


def test_embedding_rows_can_filter_by_recording(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    other = replace(recording, recording_id="other", content_hash="other-hash")
    other_windows = [
        replace(window, window_id=f"other-{window.window_id}", recording_id="other")
        for window in windows
    ]
    other_array = tmp_path / "other.npy"
    other_array.write_bytes(array_path.read_bytes())
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        for record, selected, path in [
            (recording, windows, array_path),
            (other, other_windows, other_array),
        ]:
            catalog.add_recording(record)
            catalog.register_artifact(selected, "config", path, audio_paths)
        catalog.register_artifact(windows, "second-config", array_path, audio_paths)
        all_rows = catalog.embedding_rows("config")
        assert len(all_rows) == 4
        for recording_id in ("r", "other"):
            assert catalog.embedding_rows("config", recording_id=recording_id) == [
                row for row in all_rows if row.recording_id == recording_id
            ]
        assert catalog.embedding_rows("config", recording_id="missing") == []
        assert catalog.embedding_rows("missing", recording_id="r") == []


def test_recording_aliases_resolve_csv_labels_and_survive_reopening(tmp_path, artifact):
    recording, _, _, _ = artifact
    alias = replace(recording, path=tmp_path / "copy.wav")
    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as catalog:
        catalog.add_recording(recording)
        catalog.add_recording(alias)
        assert len(catalog.recordings()) == 1
        assert {r.path for r in catalog.recording_aliases()} == {
            recording.path,
            alias.path,
        }

    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "audiofilename,start,end,label:species\nsource.wav,0,5,A\ncopy.wav,5,10,B\n"
    )
    with Catalog(path) as catalog:
        labels = read_interval_labels(csv_path, catalog.recording_aliases(), (tmp_path,))
        assert [label.recording_id for label in labels] == ["r", "r"]


def test_reregistering_artifact_can_change_its_row_order(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        catalog.register_artifact(windows, "config", array_path, audio_paths)
        catalog.register_artifact(windows[::-1], "config", array_path, audio_paths[::-1])
        assert [(row.window_id, row.row_index) for row in catalog.embedding_rows("config")] == [
            ("w0", 1),
            ("w1", 0),
        ]


def test_failed_reregistration_restores_all_previous_mappings(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        for config_id in ["config", "other"]:
            catalog.register_artifact(windows, config_id, array_path, audio_paths)
        original = catalog.embedding_rows("config")
        other = catalog.embedding_rows("other")
        catalog.connection.executescript("""
            CREATE TRIGGER fail_registration BEFORE INSERT ON window_embeddings
            WHEN NEW.window_id='w0' AND NEW.config_id='config'
            BEGIN SELECT RAISE(ABORT, 'injected write failure'); END;
        """)
        with pytest.raises(sqlite3.IntegrityError, match="injected write failure"):
            catalog.register_artifact(windows[::-1], "config", array_path, audio_paths[::-1])
        assert catalog.embedding_rows("config") == original
        assert catalog.embedding_rows("other") == other


def test_changed_path_moves_alias_without_overwriting_old_content_identity(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    changed = replace(recording, recording_id="new", content_hash="new-hash")
    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as catalog:
        catalog.add_recording(recording)
        catalog.register_artifact(windows, "config", array_path, audio_paths)
        catalog.add_recording(changed)
        assert catalog.recording_aliases() == [changed]
        assert catalog.recordings() == [changed]
        assert {row.recording_id for row in catalog.stored_recordings()} == {"r", "new"}
        assert len(catalog.windows()) == 2

    with Catalog(path) as catalog:
        assert catalog.recording_aliases() == [changed]
        assert {row.recording_id for row in catalog.stored_recordings()} == {"r", "new"}
        assert len(catalog.windows()) == 2


def test_recording_identity_rejects_conflicting_metadata(tmp_path, artifact):
    recording, _, _, _ = artifact
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        with pytest.raises(ValueError, match="conflicting metadata"):
            catalog.add_recording(replace(recording, content_hash="different"))


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((1, 3)),
        np.zeros(2),
        np.zeros((2, 0)),
        np.array([[1, float("nan")], [2, 3]]),
        np.array([["a", "b"], ["c", "d"]]),
        np.ones((2, 3), dtype=np.complex64),
    ],
)
def test_invalid_array_cannot_register_partial_windows(tmp_path, artifact, array):
    recording, windows, array_path, audio_paths = artifact
    np.save(array_path, array)
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        with pytest.raises(ValueError, match="Embedding array"):
            catalog.register_artifact(windows, "config", array_path, audio_paths)
        assert not catalog.windows()


def test_missing_audio_cannot_register_partial_windows(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    audio_paths[1].unlink()
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        with pytest.raises((OSError, RuntimeError)):
            catalog.register_artifact(windows, "config", array_path, audio_paths)
        assert not catalog.windows()


def test_short_audio_and_invalid_bounds_rejected(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        bad_windows = [replace(windows[0], start_sample=-1), windows[1]]
        with pytest.raises(ValueError, match="window bounds"):
            catalog.register_artifact(bad_windows, "config", array_path, audio_paths)
        sf.write(audio_paths[1], np.zeros(49), 10)
        with pytest.raises(ValueError, match="five-second mono"):
            catalog.register_artifact(windows, "config", array_path, audio_paths)
        assert not catalog.windows()


def test_existing_window_identity_cannot_change_policy(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        catalog.register_artifact(windows, "config", array_path, audio_paths)
        changed = [windows[0], replace(windows[1], policy="different")]
        with pytest.raises(ValueError, match="conflicting metadata"):
            catalog.register_artifact(changed, "new-config", array_path, audio_paths)
        assert not catalog.embedding_rows("new-config")
        assert catalog.windows() == windows


def test_registration_requires_cloud_audio_and_unknown_windows_fail(tmp_path, artifact):
    recording, windows, array_path, audio_paths = artifact
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.add_recording(recording)
        catalog.register_artifact(windows, "config", array_path, audio_paths)
        with pytest.raises(ValueError, match="uploaded audio"):
            catalog.mark_registered("w0")
        for uri in ["file:///clip.wav", "gs://bucket", "gs:///clip.wav"]:
            with pytest.raises(ValueError, match="gs://"):
                catalog.mark_uploaded("w0", uri)
        with pytest.raises(ValueError, match="Unknown window"):
            catalog.mark_uploaded("missing", "gs://bucket/clip.wav")
        catalog.mark_uploaded("w0", "gs://bucket/w0.wav")
        catalog.mark_registered("w0")
        assert [window.window_id for window in catalog.pending_windows()] == ["w1"]
