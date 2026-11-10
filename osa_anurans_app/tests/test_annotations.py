import csv

import pytest
from osa_anurans_app.src.annotations import (
    aggregate_votes,
    load_reference_seed_pairs,
    load_snapshot,
    map_intervals_to_windows,
    save_snapshot,
)
from osa_anurans_app.src.records import (
    IntervalLabel,
    LabelCounts,
    RecordingRecord,
    WindowRecord,
)
from osa_anurans_app.src.sources import discover_recordings, read_interval_labels


def test_reference_seed_provenance_ignores_other_annotators_and_later_votes(tmp_path):
    answers = [
        {
            "window_id": "w1",
            "query_species_id": "A",
            "annotator_id": "new-species",
            "response": "present",
        },
        {
            "window_id": "w2",
            "query_species_id": "B",
            "annotator_id": "eoin-template-seed",
            "response": "present",
        },
        {"window_id": "w1", "query_species_id": "A", "annotator_id": "human", "response": "absent"},
        {
            "window_id": "w3",
            "query_species_id": "A",
            "annotator_id": "grillos-seed-gt",
            "response": "present",
        },
        {
            "window_id": "w4",
            "query_species_id": "A",
            "annotator_id": "human",
            "response": "present",
        },
    ]
    snapshot = save_snapshot(
        tmp_path / "bq", aggregate_votes(answers), "bigquery", raw_answers=answers
    )
    assert load_reference_seed_pairs(snapshot) == {("w1", "A"), ("w2", "B")}
    csv = save_snapshot(tmp_path / "csv", [LabelCounts("w1", "A", 1)], "labels.csv")
    assert load_reference_seed_pairs(csv) == set()


def write_annotations(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audiofilename", "start", "end", "label:species"])
        writer.writerows(rows)


def recording(path, recording_id="recording", frames=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return RecordingRecord(recording_id, path, "hash", 10, frames)


def test_discovery_deduplicates_nested_roots_and_ignores_non_audio(tmp_path):
    recording(tmp_path / "nested" / "call.WAV")
    (tmp_path / "notes.txt").touch()
    assert discover_recordings((tmp_path, tmp_path / "nested")) == [
        (tmp_path / "nested" / "call.WAV").resolve()
    ]


def test_csv_resolves_relative_paths_without_colliding_basenames(tmp_path):
    first = recording(tmp_path / "a" / "same.wav", "first")
    second = recording(tmp_path / "b" / "same.wav", "second")
    csv_path = tmp_path / "annotations.csv"
    write_annotations(csv_path, [["b/same.wav", 0, 5, "DENEBR"]])
    labels = read_interval_labels(csv_path, [first, second], (tmp_path,), {"DENEBR"})
    assert labels == [IntervalLabel("second", "DENEBR", 0, 5)]


def test_csv_rejects_ambiguous_root_relative_filename(tmp_path):
    first = recording(tmp_path / "a" / "same.wav", "first")
    second = recording(tmp_path / "b" / "same.wav", "second")
    csv_path = tmp_path / "annotations.csv"
    write_annotations(csv_path, [["same.wav", 0, 5, "DENEBR"]])
    with pytest.raises(ValueError, match="multiple recording roots"):
        read_interval_labels(csv_path, [first, second], (first.path.parent, second.path.parent))


@pytest.mark.parametrize(
    "start,end", [(float("nan"), 5), (0, float("inf")), (-1, 5), (5, 5), (0, 11)]
)
def test_csv_rejects_invalid_interval_bounds(tmp_path, start, end):
    audio = recording(tmp_path / "audio.wav")
    csv_path = tmp_path / "annotations.csv"
    write_annotations(csv_path, [["audio.wav", start, end, "DENEBR"]])
    with pytest.raises(ValueError, match="annotations.csv:2"):
        read_interval_labels(csv_path, [audio], (tmp_path,))


def test_csv_rejects_missing_recordings_and_unknown_species(tmp_path):
    audio = recording(tmp_path / "audio.wav")
    csv_path = tmp_path / "annotations.csv"
    write_annotations(csv_path, [["missing.wav", 0, 5, "DENEBR"]])
    with pytest.raises(ValueError, match="not a registered recording"):
        read_interval_labels(csv_path, [audio], (tmp_path,))
    write_annotations(csv_path, [["audio.wav", 0, 5, "DENEBR"]])
    with pytest.raises(ValueError, match="Unknown species"):
        read_interval_labels(csv_path, [audio], (tmp_path,), {"LEPFRA"})


def test_all_votes_count_unsure_ignored_and_ties_unknown():
    responses = ["present", "present", "absent", "unsure"]
    rows = [
        {
            "window_id": "w",
            "query_species_id": "A",
            "response": response,
            "annotator_id": "same-annotator",
        }
        for response in responses
    ]
    rows += [
        {"window_id": "w", "query_species_id": "B", "response": response}
        for response in ["present", "absent"]
    ]
    rows += [{"window_id": "other", "query_species_id": "A", "response": "unsure"}]
    labels = {(label.window_id, label.species_id): label for label in aggregate_votes(rows)}
    assert labels[("w", "A")] == LabelCounts("w", "A", 2, 1)
    assert labels[("w", "A")].response == "present"
    assert labels[("w", "B")].response is None
    assert labels[("other", "A")].response is None


def test_interval_overlap_labels_only_present_and_warns_partial(caplog):
    windows = [WindowRecord(f"w{i}", "r", i * 50, (i + 1) * 50, 10) for i in range(3)]
    labels = map_intervals_to_windows(
        [
            IntervalLabel("r", "A", 0, 5),
            IntervalLabel("r", "A", 0, 5),
            IntervalLabel("r", "B", 4, 6),
        ],
        windows,
    )
    assert labels == [
        LabelCounts("w0", "A", 1),
        LabelCounts("w0", "B", 1),
        LabelCounts("w1", "B", 1),
    ]
    assert len(caplog.records) == 2
    assert "Partial annotation overlap" in caplog.text
    assert "overlap=1.0s" in caplog.text


def test_unmatched_interval_is_reported(caplog):
    assert map_intervals_to_windows([IntervalLabel("missing", "A", 0, 5)], []) == []
    assert "no matching window" in caplog.text


def test_snapshot_round_trip_preserves_counts_and_provenance(tmp_path):
    path = tmp_path / "annotations" / "snapshot"
    labels = [LabelCounts("w", "A", 2, 1), LabelCounts("w", "B", 1, 1)]
    save_snapshot(
        path,
        labels,
        "bigquery",
        raw_answers=[{"window_id": "w", "response": "present"}],
        windows=[WindowRecord("w", "r", 0, 50, 10)],
    )
    assert load_snapshot(path) == labels
    assert (path / "answers.jsonl").is_file()
    assert (path / "windows.csv").is_file()
    with pytest.raises(FileExistsError):
        save_snapshot(path, [], "initial_csv")


def test_invalid_snapshot_does_not_publish_partial_files(tmp_path):
    path = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="nonnegative"):
        save_snapshot(path, [LabelCounts("w", "A", -1, 0)], "test")
    assert not path.exists()


def test_snapshot_rejects_duplicate_pairs(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        save_snapshot(
            tmp_path / "snapshot",
            [LabelCounts("w", "A", 1), LabelCounts("w", "A", 2)],
            "test",
        )
