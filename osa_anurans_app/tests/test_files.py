"""Check strict JSON serialization and preservation of existing output on failure."""

import json
from pathlib import Path

import pytest
from osa_anurans_app.src.files import write_json


def test_json_writer_creates_parents_and_writes_readable_json(tmp_path):
    path = tmp_path / "nested" / "report.json"
    write_json(path, {"count": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"count": 3}
    assert path.read_bytes().endswith(b"\n")
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Path("audio.wav")])
def test_invalid_json_preserves_existing_file(tmp_path, value):
    path = tmp_path / "model.json"
    path.write_bytes(b'{"original": true}')
    with pytest.raises((ValueError, TypeError)):
        write_json(path, {"value": value})
    assert path.read_bytes() == b'{"original": true}'
    assert not path.with_suffix(".json.tmp").exists()


def test_path_stringification_requires_explicit_opt_in(tmp_path):
    path = tmp_path / "report.json"
    audio = Path("audio.wav")
    write_json(path, {"audio": audio}, default=str)
    assert json.loads(path.read_text()) == {"audio": str(audio)}


def test_failed_temporary_write_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    path.write_bytes(b'{"original": true}')

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="disk full"):
        write_json(path, {"replacement": True})
    assert path.read_bytes() == b'{"original": true}'
