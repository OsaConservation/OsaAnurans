"""Fail cheap on bad labels and keep cached actions independent of raw sources."""

import csv
import json

import numpy as np
import pytest
import soundfile as sf
from osa_anurans_app.src.audio import SAMPLE_RATE, WINDOW_SAMPLES
from osa_anurans_app.src.bacpipe_adapter import BacpipePerch
from osa_anurans_app.src.catalog import Catalog
from osa_anurans_app.src.cli import main
from osa_anurans_app.src.config import AppConfig, load_config, validate_import_sources
from osa_anurans_app.src.pipeline import import_local, refresh_annotations


def make_config(tmp_path, *, filename="recording.wav", start=0, end=5):
    root = tmp_path / "Grillos"
    root.mkdir()
    sf.write(root / "recording.wav", np.zeros(WINDOW_SAMPLES), SAMPLE_RATE)
    annotations = root / "annotations.csv"
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["audiofilename", "start", "end", "label:species"]
        )
        writer.writeheader()
        writer.writerow(
            {"audiofilename": filename, "start": start, "end": end, "label:species": "A"}
        )
    return AppConfig((root,), annotations, tmp_path / "artifacts")


@pytest.mark.parametrize(
    "filename,start,end,expected",
    [
        ("missing.wav", 0, 5, "unavailable audio"),
        ("recording.wav", 0, 6, "after the recording"),
        ("recording.wav", -1, 5, "Invalid interval bounds"),
    ],
)
def test_bad_annotations_fail_before_backend_initialization(
    tmp_path, monkeypatch, filename, start, end, expected
):
    config = make_config(tmp_path, filename=filename, start=start, end=end)

    def forbidden_init(*args, **kwargs):
        raise AssertionError("Model initialization must not happen for invalid labels")

    monkeypatch.setattr(BacpipePerch, "__init__", forbidden_init)
    with pytest.raises(ValueError, match=expected):
        import_local(config)
    reports = list((config.artifacts / "imports").glob("*/import.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["state"] == "annotation_preflight_failed"
    assert report["generated"] == 0
    excluded = reports[0].parent / "annotations" / "excluded_annotations.json"
    assert excluded.exists()
    if filename == "missing.wav":
        assert json.loads(excluded.read_text())[0]["audiofilename"] == filename


def test_preflight_does_not_resolve_current_missing_audio_against_stale_catalog(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    old_root = tmp_path / "old"
    old_root.mkdir()
    stale = old_root / "missing.wav"
    sf.write(stale, np.zeros(WINDOW_SAMPLES), SAMPLE_RATE)
    from osa_anurans_app.src.audio import identify_recording

    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        catalog.add_recording(identify_recording(stale))
    # A matching historical alias is not part of this run's configured inputs.
    config.annotations_csv.write_text(
        f"audiofilename,start,end,label:species\n{stale},0,5,A\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        BacpipePerch, "__init__", lambda *args, **kwargs: pytest.fail("backend initialized")
    )
    with pytest.raises(ValueError, match="unavailable audio"):
        import_local(config)


def test_cached_status_does_not_require_original_audio_or_csv(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[paths]\naudio_roots = ['removed-recordings']\n"
        "annotations_csv = 'removed-annotations.csv'\nartifacts = 'artifacts'\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert main(["--config", str(config_path), "status"]) == 0
    assert "Recordings: 0; windows: 0" in capsys.readouterr().out
    with pytest.raises(ValueError, match="existing directory"):
        validate_import_sources(config)
    assert main(["--config", str(config_path), "import"]) == 1


def test_config_allows_an_audio_inbox_without_csv(tmp_path):
    inbox = tmp_path / "audio-inbox"
    inbox.mkdir()
    path = tmp_path / "config.toml"
    path.write_text("[paths]\naudio_roots = ['audio-inbox']\n", encoding="utf-8")
    config = load_config(path)
    assert config.audio_roots == (inbox,)
    assert config.annotations_csv is None
    validate_import_sources(config)


def test_bigquery_refresh_reports_nonlocal_present_absent_and_unsure(tmp_path, caplog):
    config = make_config(tmp_path)
    answers = [
        {
            "window_id": name,
            "query_species_id": "A",
            "response": response,
            "recording_id": "cloud-recording",
            "start_seconds": 30,
            "end_seconds": 35,
            "audio_gcs_uri": "gs://bucket/audio.wav",
        }
        for name, response in [
            ("unknown-present", "present"),
            ("unknown-absent", "absent"),
            ("unknown-unsure", "unsure"),
        ]
    ]

    class BigQuery:
        def fetch_annotations(self):
            return answers

    snapshot = refresh_annotations(config, BigQuery())
    unmatched = json.loads((snapshot / "unmatched_windows.json").read_text())
    assert unmatched == {
        "window_ids": ["unknown-absent", "unknown-present", "unknown-unsure"],
        "answer_count": 3,
    }
    assert "3 window IDs absent from the local catalog" in caplog.text
    restored = [json.loads(line) for line in (snapshot / "answers.jsonl").read_text().splitlines()]
    assert restored == answers
