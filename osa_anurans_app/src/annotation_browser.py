"""Read saved annotation rows and local media without changing pipeline state."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .catalog import Catalog
from .config import AppConfig


@dataclass(frozen=True)
class BrowserData:
    payload: dict
    media_paths: dict[str, Path]


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _snapshot_path(config: AppConfig, snapshot: Path | None) -> Path:
    if snapshot is not None:
        return snapshot.resolve()
    latest = config.artifacts / "latest.json"
    if not latest.is_file():
        raise ValueError("No saved annotation snapshot. Run the 'refresh' command first.")
    selected = _read_json(latest).get("snapshot")
    if not isinstance(selected, str) or not selected:
        raise ValueError("No annotation snapshot selected. Run 'refresh' or use --snapshot.")
    return Path(selected).resolve()


def _saved_windows(snapshot: Path) -> dict[str, dict]:
    path = snapshot / "windows.csv"
    if not path.is_file():
        return {}
    windows = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                rate = int(row["sample_rate"])
                if rate <= 0:
                    raise ValueError("sample_rate must be positive")
                windows[row["window_id"]] = {
                    "recording_id": row["recording_id"],
                    "start_seconds": int(row["start_sample"]) / rate,
                    "end_seconds": int(row["end_sample"]) / rate,
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid window metadata in {path}:{number}: {exc}") from exc
    return windows


def _optional_text(value):
    return value if isinstance(value, str) and value.strip() else None


def _read_answers(path: Path) -> list[dict]:
    answers = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an annotation object")
                for field in ("window_id", "query_species_id"):
                    if not _optional_text(row.get(field)):
                        raise ValueError(f"missing {field}")
                if row.get("response") not in {"present", "absent", "unsure"}:
                    raise ValueError("response must be present, absent, or unsure")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid annotation in {path}:{number}: {exc}") from exc
            answers.append({**row, "row_id": str(number)})
    return answers


def load_browser_data(config: AppConfig, snapshot: Path | None = None) -> BrowserData:
    """Join every raw answer with optional local context; never aggregate votes."""
    directory = _snapshot_path(config, snapshot)
    metadata = _read_json(directory / "snapshot.json")
    if metadata.get("version") != 1:
        raise ValueError(f"Unsupported annotation snapshot version in {directory}")
    payload = {
        "snapshot": {
            "path": str(directory),
            "created_at": metadata.get("created_at"),
            "source": metadata.get("source"),
        },
        "annotations": [],
    }
    answer_path = directory / "answers.jsonl"
    if not answer_path.is_file():
        payload["message"] = (
            "Individual annotation rows are unavailable in this snapshot. "
            "Run the 'refresh' command, then restart the annotation browser."
        )
        return BrowserData(payload, {})
    answers = _read_answers(answer_path)
    windows = _saved_windows(directory)
    names = {}
    paths = {}
    catalog_path = config.artifacts / "catalog.sqlite"
    if catalog_path.is_file():
        with Catalog(catalog_path, read_only=True) as catalog:
            windows.update({
                window.window_id: {
                    "recording_id": window.recording_id,
                    "start_seconds": window.start_seconds,
                    "end_seconds": window.end_seconds,
                }
                for window in catalog.windows()
            })
            names = {row.recording_id: row.path.name for row in catalog.stored_recordings()}
            names.update({row.recording_id: row.path.name for row in catalog.recordings()})
            paths = catalog.audio_paths()
    media_paths = {}
    for answer in answers:
        window_id = answer["window_id"]
        context = windows.get(window_id, {})
        path = paths.get(window_id)
        if path is not None and path.is_file():
            media_paths[window_id] = path
        payload["annotations"].append({
            "row_id": answer["row_id"],
            "annotation_id": _optional_text(answer.get("annotation_id")),
            "window_id": window_id,
            "species_id": answer["query_species_id"],
            "response": answer["response"],
            "annotator_name": _optional_text(answer.get("annotator_name")),
            "annotator_id": _optional_text(answer.get("annotator_id")),
            "created_at": _optional_text(answer.get("created_at")),
            "recording_id": context.get("recording_id"),
            "recording_filename": names.get(context.get("recording_id")),
            "start_seconds": context.get("start_seconds"),
            "end_seconds": context.get("end_seconds"),
            "audio_url": f"/media/{quote(window_id, safe='')}.wav"
            if window_id in media_paths else None,
        })
    return BrowserData(payload, media_paths)
