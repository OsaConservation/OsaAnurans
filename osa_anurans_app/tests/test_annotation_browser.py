import json
import sqlite3

import pytest
from osa_anurans_app.src.annotation_browser import load_browser_data
from osa_anurans_app.src.catalog import Catalog
from osa_anurans_app.src.config import AppConfig


@pytest.fixture
def saved(tmp_path):
    config = AppConfig(audio_roots=(tmp_path,), annotations_csv=None, artifacts=tmp_path)
    snapshot = tmp_path / "annotations" / "saved"
    snapshot.mkdir(parents=True)
    (snapshot / "snapshot.json").write_text(json.dumps({
        "version": 1, "created_at": "2026-11-10T00:00:00Z", "source": "bigquery",
    }))
    (tmp_path / "latest.json").write_text(json.dumps({"snapshot": str(snapshot)}))
    return config, snapshot


def write_answers(snapshot, rows):
    (snapshot / "answers.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )


def answer(**fields):
    return {"window_id": "w1", "query_species_id": "FROG", "response": "present", **fields}


def test_preserves_repeated_rows_and_missing_nullable_fields(saved):
    config, snapshot = saved
    write_answers(snapshot, [answer(annotation_id="same"), answer(annotation_id="same"),
                             answer(window_id="missing", response="unsure")])
    data = load_browser_data(config)
    rows = data.payload["annotations"]
    assert [row["row_id"] for row in rows] == ["1", "2", "3"]
    assert [row["response"] for row in rows] == ["present", "present", "unsure"]
    assert all(row["audio_url"] is None for row in rows)
    assert rows[0]["annotator_name"] is None
    assert not (config.artifacts / "catalog.sqlite").exists()


def test_joins_media_without_modifying_catalog_or_latest(saved):
    config, snapshot = saved
    write_answers(snapshot, [answer(), answer(window_id="w2", response="absent")])
    wav = config.artifacts / "clip.wav"
    wav.write_bytes(b"test audio")
    database = config.artifacts / "catalog.sqlite"
    with Catalog(database) as catalog:
        catalog.connection.execute(
            "INSERT INTO recordings VALUES ('r1', 'recording.wav', 'hash', 32000, 320000)"
        )
        catalog.connection.execute(
            "INSERT INTO windows VALUES ('w1','r1',160000,320000,32000,'policy',?,0)",
            (str(wav),),
        )
        catalog.connection.commit()
    before = database.read_bytes()
    latest_before = (config.artifacts / "latest.json").read_bytes()
    data = load_browser_data(config, snapshot)
    row = data.payload["annotations"][0]
    assert row["recording_filename"] == "recording.wav"
    assert (row["start_seconds"], row["end_seconds"]) == (5, 10)
    assert row["audio_url"] == "/media/w1.wav"
    assert data.media_paths == {"w1": wav}
    assert database.read_bytes() == before
    assert (config.artifacts / "latest.json").read_bytes() == latest_before


def test_saved_window_context_without_catalog(saved):
    config, snapshot = saved
    write_answers(snapshot, [answer()])
    (snapshot / "windows.csv").write_text(
        "window_id,recording_id,start_sample,end_sample,sample_rate,policy\n"
        "w1,r1,160000,320000,32000,policy\n"
    )
    row = load_browser_data(config).payload["annotations"][0]
    assert row["recording_id"] == "r1"
    assert row["start_seconds"] == 5
    assert row["audio_url"] is None


def test_missing_and_empty_answers_are_distinct(saved):
    config, snapshot = saved
    assert "refresh" in load_browser_data(config).payload["message"]
    write_answers(snapshot, [])
    assert load_browser_data(config).payload["annotations"] == []
    assert "message" not in load_browser_data(config).payload


@pytest.mark.parametrize("line", ['{', '[]', '{"response":"invalid"}'])
def test_malformed_answers_include_line_context(saved, line):
    config, snapshot = saved
    (snapshot / "answers.jsonl").write_text("\n" + line)
    with pytest.raises(ValueError, match=r"answers.jsonl:2"):
        load_browser_data(config)


def test_explicit_snapshot_ignores_latest(saved):
    config, snapshot = saved
    write_answers(snapshot, [answer()])
    (config.artifacts / "latest.json").write_text("invalid JSON")
    assert len(load_browser_data(config, snapshot).payload["annotations"]) == 1


def test_read_only_catalog_rejects_writes_and_does_not_create(tmp_path):
    path = tmp_path / "catalog.sqlite"
    with Catalog(path):
        pass
    before = path.read_bytes()
    with Catalog(path, read_only=True) as catalog:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            catalog.connection.execute("INSERT INTO recordings VALUES ('r','p','h',1,1)")
    assert path.read_bytes() == before
    missing = tmp_path / "missing" / "catalog.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        Catalog(missing, read_only=True)
    assert not missing.parent.exists()


def test_missing_latest_has_actionable_error(tmp_path):
    config = AppConfig(audio_roots=(tmp_path,), annotations_csv=None, artifacts=tmp_path)
    with pytest.raises(ValueError, match="refresh"):
        load_browser_data(config)
