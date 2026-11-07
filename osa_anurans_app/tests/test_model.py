"""Behavioral checks for grouped training and species-specific candidates."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from bacpipe.embedding_evaluation.probing.train_probe import LinearProbe as BacpipeLinearProbe
from osa_anurans_app.src.candidates import iter_candidates
from osa_anurans_app.src.config import AppConfig
from osa_anurans_app.src.model.dataloader import EmbeddingStore
from osa_anurans_app.src.model.datasets import build_dataset, split_by_recording
from osa_anurans_app.src.model.model import LinearProbe, ProbeArtifact, load_artifact
from osa_anurans_app.src.model.train import train_classifier
from osa_anurans_app.src.records import EmbeddingRow, LabelCounts
from osa_anurans_app.src.scoring import score_candidate


def examples(tmp_path, recording_count=5):
    rows = []
    labels = []
    for recording in range(recording_count):
        path = tmp_path / f"r{recording}.npy"
        np.save(path, np.array([[2, 0], [0, 2], [1, 1]], dtype=np.float32))
        for index in range(3):
            window = f"r{recording}w{index}"
            rows.append(EmbeddingRow(window, f"r{recording}", "perch-config", path, index))
            if index < 2:
                labels.append(LabelCounts(window, ["A", "B"][index], present=1))
    return rows, labels


def settings(tmp_path):
    return AppConfig(
        (tmp_path,),
        tmp_path / "annotations.csv",
        tmp_path,
        device="cpu",
        epochs=30,
        learning_rate=0.1,
    )


def test_single_label_filtering_and_recording_split(tmp_path):
    rows, labels = examples(tmp_path)
    labels.extend(
        [
            LabelCounts(
                "r0w0", "B", present=1
            ),  # Known co-occurrence cannot train a single-label model.
            LabelCounts("r0w2", "A", present=1, absent=1),  # Tie is unknown.
            LabelCounts("r1w2", "A", absent=3),
        ]
    )
    dataset = build_dataset(rows, labels)
    assert len(dataset.rows) == 9
    assert dataset.excluded == {"unknown_or_absent_only": 5, "multiple_present_species": 1}
    split = split_by_recording(dataset, seed=42)
    assert split == split_by_recording(dataset, seed=42)
    train_recordings = {dataset.rows[i].recording_id for i in split.train}
    test_recordings = {dataset.rows[i].recording_id for i in split.test}
    assert not train_recordings & test_recordings
    assert len(test_recordings) == 1
    assert set(split.train + split.test) == set(range(len(dataset.rows)))


def test_probe_training_reload_and_full_candidate_coverage(tmp_path, caplog):
    caplog.set_level("INFO")
    rows, labels = examples(tmp_path)
    dataset = build_dataset(rows, labels)
    output = tmp_path / "run-1"
    artifact = train_classifier(dataset, output, settings(tmp_path))
    assert type(artifact.model) is BacpipeLinearProbe
    restored = load_artifact(output)
    inputs = np.array([[2, 0], [0, 2]], dtype=np.float32)
    assert np.allclose(artifact.predict_logits(inputs), restored.predict_logits(inputs))
    assert restored.species_ids == ("A", "B")
    assert restored.predict_probabilities(inputs).argmax(axis=1).tolist() == [0, 1]
    assert np.allclose(restored.predict_probabilities(inputs).sum(axis=1), 1)

    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["test"]["accuracy"] == 1
    expected_loss = torch.nn.functional.cross_entropy(
        torch.from_numpy(restored.predict_logits(inputs)), torch.tensor([0, 1])
    ).item()
    assert metrics["test"]["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert "Test set: 2 examples" in caplog.text
    with (output / "split.csv").open(newline="") as handle:
        splits = list(csv.DictReader(handle))
    assert {row["recording_id"] for row in splits if row["partition"] == "train"}.isdisjoint(
        row["recording_id"] for row in splits if row["partition"] == "test"
    )

    candidates = list(
        iter_candidates(rows, restored, batch_size=2, created_at="2026-11-06T00:00:00+00:00")
    )
    assert len(candidates) == len(rows) * 2  # Includes unlabeled and already labeled windows.
    assert len({row["candidate_id"] for row in candidates}) == len(candidates)
    assert set(candidates[0]) == {
        "candidate_id",
        "window_id",
        "query_species_id",
        "created_at",
        "score",
    }
    assert {(row["window_id"], row["query_species_id"]) for row in candidates} == {
        (row.window_id, species) for row in rows for species in artifact.species_ids
    }
    repeated = list(iter_candidates(rows, restored, created_at="same timestamp"))
    assert [row["candidate_id"] for row in candidates] == [row["candidate_id"] for row in repeated]
    assert all(0 <= row["score"] <= 1 for row in candidates)


def test_one_training_class_fails_without_model_write(tmp_path):
    rows, _ = examples(tmp_path)
    labels = [LabelCounts(row.window_id, "A", present=1) for row in rows]
    output = tmp_path / "failed-run"
    with pytest.raises(ValueError, match="at least two species"):
        train_classifier(build_dataset(rows, labels), output, settings(tmp_path))
    assert not (output / "model.pt").exists()


def test_existing_run_and_missing_embedding_are_reported(tmp_path):
    rows, labels = examples(tmp_path)
    output = tmp_path / "existing-run"
    output.mkdir()
    (output / "model.pt").write_text("existing model", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty directory"):
        train_classifier(build_dataset(rows, labels), output, settings(tmp_path))
    assert (output / "model.pt").read_text() == "existing model"

    missing = tmp_path / "missing.npy"
    with pytest.raises(ValueError, match="Cannot read embedding artifact for window r0w0"):
        EmbeddingStore().read(replace(rows[0], path=missing))


def test_rare_species_recording_stays_in_training(tmp_path):
    rows, labels = examples(tmp_path)
    dataset = build_dataset(rows, labels)
    split = split_by_recording(dataset, seed=42)
    held_out = dataset.rows[split.test[0]].recording_id
    labels = [
        replace(label, species_id="C")
        if label.species_id == "B" and label.window_id.startswith(held_out + "w")
        else label
        for label in labels
    ]
    output = tmp_path / "skip-run"
    artifact = train_classifier(
        build_dataset(rows, labels), output, replace(settings(tmp_path), epochs=1)
    )
    assert artifact.species_ids == ("A", "B", "C")
    assert json.loads((output / "skipped_species.json").read_text()) == []
    assert (
        json.loads((output / "metrics.json").read_text())["test_examples_with_untrained_species"]
        == 0
    )
    with (output / "split.csv").open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert all(row["partition"] == "train" for row in saved if row["species_id"] == "C")


def test_embedding_alignment_and_configuration_errors(tmp_path):
    rows, labels = examples(tmp_path)
    with pytest.raises(ValueError, match="multiple embedding"):
        build_dataset(rows + rows[:1], labels)
    with pytest.raises(ValueError, match="one embedding configuration"):
        build_dataset(rows + [replace(rows[0], config_id="other")], labels)
    with pytest.raises(ValueError, match="same embedding file row"):
        build_dataset(rows + [replace(rows[0], window_id="different-window")], labels)
    with pytest.raises(ValueError, match="Duplicate consensus"):
        build_dataset(rows, labels + labels[:1])
    with pytest.raises(ValueError, match="at least two recordings"):
        split_by_recording(build_dataset(rows[:3], labels))

    np.save(tmp_path / "bad.npy", np.array([[np.nan, 1]]))
    with pytest.raises(ValueError, match="nonfinite"):
        EmbeddingStore().read(replace(rows[0], path=tmp_path / "bad.npy"))
    with pytest.raises(ValueError, match="Invalid embedding location"):
        EmbeddingStore().read(replace(rows[0], row_index=99))


def test_scores_are_probability_based_and_raw_logit_shift_invariant():
    model = LinearProbe(2, 2)
    with torch.no_grad():
        model.probe.weight.zero_()
        model.probe.bias[:] = torch.tensor([2.0, 1.0])
    artifact = ProbeArtifact(model, ("A", "B"), "config", "run")
    inputs = np.zeros((1, 2), dtype=np.float32)
    original = artifact.predict_probabilities(inputs)
    with torch.no_grad():
        model.probe.bias += 100
    assert np.allclose(original, artifact.predict_probabilities(inputs))
    assert score_candidate(0.5) == 0
    assert score_candidate(0) == score_candidate(1) == 1
    assert score_candidate(0.4) == pytest.approx(score_candidate(0.6))
    for value in [np.nan, np.inf, -0.1, 1.1]:
        with pytest.raises(ValueError):
            score_candidate(value)


def test_candidate_config_and_duplicate_windows_rejected(tmp_path):
    rows, _ = examples(tmp_path)
    artifact = ProbeArtifact(LinearProbe(2, 2), ("A", "B"), "other", "run")
    with pytest.raises(ValueError, match="configuration"):
        list(iter_candidates(rows, artifact))
    artifact.config_id = "perch-config"
    with pytest.raises(ValueError, match="Duplicate"):
        list(iter_candidates(rows + rows[:1], artifact))


def test_split_keeps_all_species_when_no_holdout_is_possible(tmp_path, caplog):
    rows, _ = examples(tmp_path, recording_count=2)
    labels = [LabelCounts(row.window_id, row.recording_id, present=1) for row in rows]
    split = split_by_recording(build_dataset(rows, labels))
    assert split.train == list(range(len(rows)))
    assert split.test == []
    assert "held-out evaluation is unavailable" in caplog.text
    output = tmp_path / "all-training"
    artifact = train_classifier(
        build_dataset(rows, labels), output, replace(settings(tmp_path), epochs=1)
    )
    assert artifact.species_ids == ("r0", "r1")
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["held_out_evaluation_available"] is False
    assert metrics["test"]["examples"] == 0
    assert metrics["test"]["accuracy"] is None
    assert metrics["test"]["loss"] is None


def test_split_reduces_holdout_to_preserve_species(tmp_path):
    rows, _ = examples(tmp_path, recording_count=4)
    labels = [
        LabelCounts(
            row.window_id,
            "shared" if row.recording_id in {"r0", "r1"} else row.recording_id,
            present=1,
        )
        for row in rows
    ]
    dataset = build_dataset(rows, labels)
    split = split_by_recording(dataset, test_fraction=0.75)
    assert len({dataset.rows[i].recording_id for i in split.test}) == 1
    assert {dataset.labels[i] for i in split.train} == set(dataset.species_ids)
