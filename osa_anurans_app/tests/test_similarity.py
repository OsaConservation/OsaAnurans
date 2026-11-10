"""Verify exact retrieval and the mixed queue policy with known vectors."""

import json
from dataclasses import replace

import numpy as np
import pytest
from osa_anurans_app.src.candidates import select_pools, write_candidates
from osa_anurans_app.src.cloud.bigquery import validate_candidates
from osa_anurans_app.src.config import AppConfig, validate_config
from osa_anurans_app.src.records import EmbeddingRow, LabelCounts
from osa_anurans_app.src.scoring import similarity_fraction
from osa_anurans_app.src.similarity import rank_similar_windows


def vectors(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(path, np.array([[1, 0], [1, 1], [-1, 0], [0, 1], [0, 0]], dtype=np.float32))
    return [EmbeddingRow(f"w{i}", f"r{i}", "config", path, i) for i in range(5)]


@pytest.mark.parametrize("count,expected", [(0, 1), (5, 25 / 30), (25, 0.5), (100, 0.2)])
def test_formula(count, expected):
    assert similarity_fraction(count) == expected


@pytest.mark.parametrize("balance", [0, -1, float("nan"), float("inf"), True])
def test_invalid_balance(tmp_path, balance):
    with pytest.raises(ValueError):
        similarity_fraction(1, balance)
    with pytest.raises(ValueError):
        validate_config(replace(AppConfig((tmp_path,), None, tmp_path), similarity_balance=balance))


def test_cosine_is_exact_batched_and_excludes_seeds_and_zero_vectors(tmp_path):
    rows = vectors(tmp_path)
    for size in [1, 2, 10]:
        result = rank_similar_windows(list(reversed(rows)), [rows[0]], size)
        assert [window for window, _ in result] == ["w1", "w3", "w2"]
        assert np.allclose([score for _, score in result], [1 / np.sqrt(2), 0, -1])
    assert rank_similar_windows(rows, [rows[4]]) == []
    with pytest.raises(ValueError, match="configuration"):
        rank_similar_windows(rows, [replace(rows[0], config_id="other")])


def test_multiple_seeds_use_maximum_and_reject_nonfinite(tmp_path):
    rows = vectors(tmp_path)
    result = rank_similar_windows(rows, [rows[0], rows[2]])
    assert result[0][0] == "w1"
    assert np.isclose(result[0][1], 1 / np.sqrt(2))
    np.save(rows[0].path, np.array([[float("nan"), 1]], dtype=np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        rank_similar_windows(rows[:1], rows[:1])


def test_pool_overlap_refill_and_similarity_first():
    similar = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    uncertain = [("a", 0.01), ("d", 0.02), ("b", 0.03)]
    selected = select_pools(similar, uncertain, 4, 0.5)
    assert selected == [
        ("a", "similarity", 0.9),
        ("b", "similarity", 0.8),
        ("c", "similarity", 0.7),
        ("d", "uncertainty", 0.02),
    ]
    assert select_pools(similar[:1], uncertain, 5, 0.9) == [
        ("a", "similarity", 0.9),
        ("d", "uncertainty", 0.02),
        ("b", "uncertainty", 0.03),
    ]


def test_similarity_only_cloud_schema_multi_positive_and_test_exclusion(tmp_path):
    rows = vectors(tmp_path)
    labels = [
        LabelCounts("w0", "A", 2),
        LabelCounts("w0", "B", 1),
        LabelCounts("w2", "HELD", 1),
        LabelCounts("w1", "ABSENT", 0, 1),
    ]
    path = tmp_path / "candidates.jsonl"
    count = write_candidates(path, rows, None, 2, 10, labels=labels, held_out_recordings={"r2"})
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert count == 4
    assert {row["query_species_id"] for row in saved} == {"A", "B"}
    assert {row["window_id"] for row in saved} == {"w1", "w3"}
    validate_candidates(path, count)
    summary = json.loads((tmp_path / "candidate-selection.json").read_text())["species"]
    assert summary["A"]["positive_recordings"] == 1
    assert summary["A"]["selected_uncertainty"] == 0


def test_positive_recordings_counted_once_and_unsupported_species_uses_similarity(tmp_path):
    from osa_anurans_app.src.model.model import LinearProbe, ProbeArtifact

    rows = vectors(tmp_path)
    rows[1] = replace(rows[1], recording_id=rows[0].recording_id)
    artifact = ProbeArtifact(LinearProbe(2, 2), ("A", "B"), "config", "run")
    path = tmp_path / "candidates.jsonl"
    write_candidates(
        path,
        rows,
        artifact,
        2,
        2,
        labels=[LabelCounts("w0", "A", 1), LabelCounts("w1", "A", 1), LabelCounts("w3", "NEW", 1)],
    )
    report = json.loads((tmp_path / "candidate-selection.json").read_text())
    assert report["species"]["A"]["positive_recordings"] == 1
    assert report["species"]["A"]["positive_windows"] == 2
    assert report["species"]["A"]["similarity_fraction"] == 25 / 26
    assert report["species"]["NEW"]["similarity_fraction"] == 1
    assert report["species"]["NEW"]["selected_uncertainty"] == 0


@pytest.mark.parametrize("with_classifier", [False, True])
def test_ordinary_resolved_pairs_are_excluded_only_from_similarity(tmp_path, with_classifier):
    from osa_anurans_app.src.model.model import LinearProbe, ProbeArtifact

    rows = vectors(tmp_path)
    labels = [
        LabelCounts("w0", "A", 100),
        LabelCounts("w1", "A", 0, 1),
        LabelCounts("w3", "A", 1, 1),
        LabelCounts("w2", "B", 1),
        LabelCounts("unsure", "A", 0, 0),
    ]
    artifact = (
        ProbeArtifact(LinearProbe(2, 2), ("A", "B"), "config", "run") if with_classifier else None
    )
    path = tmp_path / "candidates.jsonl"
    write_candidates(path, rows, artifact, 2, 10, labels=labels)
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    selected_a = {
        row["window_id"]: row["selection_method"] for row in saved if row["query_species_id"] == "A"
    }
    # A label for B does not exclude A; a tied A answer remains unknown.
    assert selected_a["w2"] == "similarity"
    assert selected_a["w3"] == "similarity"
    if with_classifier:
        # With similarity exhausted, uncertainty may revisit present and absent pairs.
        assert selected_a["w0"] == "uncertainty"
        assert selected_a["w1"] == "uncertainty"
    else:
        assert set(selected_a) == {"w2", "w3"}


@pytest.mark.parametrize("with_classifier", [False, True])
def test_reference_pairs_stay_seeds_but_never_become_review_candidates(tmp_path, with_classifier):
    from osa_anurans_app.src.model.model import LinearProbe, ProbeArtifact

    rows = vectors(tmp_path)
    labels = [
        LabelCounts("w0", "A", 1),
        LabelCounts("w1", "A", 0, 1),
        LabelCounts("w3", "A", 1, 1),
        LabelCounts("w2", "B", 1),
    ]
    artifact = (
        ProbeArtifact(LinearProbe(2, 2), ("A", "B"), "config", "run") if with_classifier else None
    )
    excluded = {("w0", "A"), ("w3", "A"), ("w2", "B")}
    output = tmp_path / "candidates.jsonl"
    write_candidates(output, rows, artifact, 2, 10, labels=labels, reference_seed_pairs=excluded)
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert not excluded.intersection((row["window_id"], row["query_species_id"]) for row in saved)
    selected_a = {
        row["window_id"]: row["selection_method"] for row in saved if row["query_species_id"] == "A"
    }
    assert selected_a["w2"] == "similarity"  # Another species' reference is still eligible.
    if with_classifier:
        assert selected_a["w1"] == "uncertainty"  # Ordinary answers can be revisited.
    summary = json.loads((tmp_path / "candidate-selection.json").read_text())["species"]
    assert summary["A"]["positive_windows"] == 1
