"""Check mixed queue priorities and compatibility with the classifier's scores."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from osa_anurans_app.src.candidates import (
    _uncertainty_pools,
    iter_uncertainty_scores,
    select_pools,
    write_candidates,
)
from osa_anurans_app.src.config import AppConfig, load_config, validate_config
from osa_anurans_app.src.model.model import LinearProbe, ProbeArtifact
from osa_anurans_app.src.records import EmbeddingRow, LabelCounts


@pytest.mark.parametrize("limit", [2, 100])
def test_saved_candidates_have_unique_windows_and_ascending_queue_scores(tmp_path, limit):
    path = tmp_path / "embeddings.npy"
    np.save(path, np.array([[0, 0], [1, 0], [-1, 0], [0, 0]], dtype=np.float32))
    rows = [EmbeddingRow(f"w{i}", "recording", "config", path, i) for i in range(4)]
    model = LinearProbe(2, 2)
    with torch.no_grad():
        model.probe.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        model.probe.bias.zero_()
    artifact = ProbeArtifact(model, ("A", "B"), "config", "run")
    exhaustive = list(iter_uncertainty_scores(rows, artifact))
    output = tmp_path / "candidates.jsonl"
    # Reversing inputs must not change which tied windows are selected.
    count = write_candidates(
        output,
        list(reversed(rows)),
        artifact,
        3,
        limit,
        labels=[LabelCounts("w1", "A", 1), LabelCounts("w2", "B", 1)],
    )
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == len(saved) == 2 * min(limit, len(rows))
    for species in artifact.species_ids:
        actual = [row for row in saved if row["query_species_id"] == species]
        assert [row["score"] for row in actual] == [
            i / max(len(actual) - 1, 1) for i in range(len(actual))
        ]
        assert len({row["window_id"] for row in actual}) == len(actual)
        assert {(row["window_id"], row["query_species_id"]) for row in actual} <= {
            (window, species) for window, species, _ in exhaustive
        }


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_candidate_limit_is_rejected(tmp_path, limit):
    config = AppConfig((tmp_path,), tmp_path / "annotations.csv", tmp_path)
    assert config.candidates_per_species == 100
    with pytest.raises(ValueError, match="candidates_per_species"):
        validate_config(replace(config, candidates_per_species=limit))
    with pytest.raises(ValueError, match="candidates_per_species"):
        write_candidates(tmp_path / "candidates.jsonl", [], None, 4, limit, labels=[])


def test_candidate_limit_loads_from_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[paths]\naudio_roots = ['.']\nannotations_csv = 'annotations.csv'\n"
        "[candidates]\ncandidates_per_species = 3\n",
        encoding="utf-8",
    )
    assert load_config(config_path).candidates_per_species == 3


@pytest.mark.parametrize("batch_size", [1, 3])
def test_mixed_queue_selects_exact_most_uncertain_windows_and_breaks_ties(tmp_path, batch_size):
    path = tmp_path / "vectors.npy"
    # The classifier uses only x: x=0 gives probability 0.5 for both classes.
    # Four equally uncertain windows compete for three slots; IDs break the tie.
    windows = ["seed", "similar", "less-uncertain", "u-d", "u-c", "u-b", "u-a"]
    np.save(
        path, np.array([[4, 0], [3, 0], [1, 1], [0, 4], [0, 3], [0, 2], [0, 1]], dtype=np.float32)
    )
    rows = [EmbeddingRow(window, window, "config", path, i) for i, window in enumerate(windows)]
    model = LinearProbe(2, 2)
    with torch.no_grad():
        model.probe.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        model.probe.bias.zero_()
    artifact = ProbeArtifact(model, ("A", "B"), "config", "run")
    output = tmp_path / "candidates.jsonl"

    # One positive recording and balance=1/3 allocate one similarity slot out of four.
    count = write_candidates(
        output,
        list(reversed(rows)),
        artifact,
        batch_size,
        4,
        labels=[LabelCounts("seed", "A", 1)],
        similarity_balance=1 / 3,
    )
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 4
    assert [row["window_id"] for row in saved] == ["similar", "u-a", "u-b", "u-c"]
    assert [row["score"] for row in saved] == [0, 1 / 3, 2 / 3, 1]
    diagnostics = json.loads((tmp_path / "candidate-selection.json").read_text())["candidates"]
    assert [row["selection_method"] for row in diagnostics] == ["similarity"] + ["uncertainty"] * 3
    assert all(row["uncertainty"] == 0 for row in diagnostics[1:])


@pytest.mark.parametrize("limit", [1, 4, 100])
def test_uncertainty_heap_matches_exhaustive_sort_and_mixed_selection(tmp_path, monkeypatch, limit):
    path = tmp_path / "vectors.npy"
    # Repeated x values deliberately create ties across inference batches.
    rng = np.random.default_rng(42)
    np.save(
        path, np.column_stack([rng.integers(-3, 4, 60), rng.normal(size=60)]).astype(np.float32)
    )
    rows = [EmbeddingRow(f"w{i:03}", "recording", "config", path, i) for i in range(60)]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    model = LinearProbe(2, 2)
    with torch.no_grad():
        model.probe.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        model.probe.bias.zero_()
    artifact = ProbeArtifact(model, ("A", "B"), "config", "run")
    exhaustive = list(iter_uncertainty_scores(rows, artifact, batch_size=7))

    def unexpected_id(*args):
        raise AssertionError("Uncertainty ranking should not create candidate IDs")

    monkeypatch.setattr("osa_anurans_app.src.candidates.uuid5", unexpected_id)
    pools = _uncertainty_pools(rows, artifact, 7, limit)
    for species in artifact.species_ids:
        expected = sorted(
            [(window, score) for window, target, score in exhaustive if target == species],
            key=lambda item: (item[1], item[0]),
        )
        assert pools[species] == expected[: 2 * limit]
        # Worst-case overlap: similarity takes the most uncertain windows first.
        similar = [(window, 1 - i / len(expected)) for i, (window, _) in enumerate(expected)]
        for fraction in [0, 0.25, 0.9, 1]:
            assert select_pools(similar, pools[species], limit, fraction) == select_pools(
                similar, expected, limit, fraction
            )
        # Similarity shortages must still refill from uncertainty without changing results.
        assert select_pools(similar[:1], pools[species], limit, 0.9) == select_pools(
            similar[:1], expected, limit, 0.9
        )


def test_uncertainty_exclusions_apply_before_heap_retention_and_only_to_target_species(monkeypatch):
    scores = [(f"known-{i}", "A", 0) for i in range(20)] + [
        ("fresh-a", "A", 0.1),
        ("fresh-b", "A", 0.2),
        ("fresh-c", "A", 0.3),
        ("known-0", "B", 0),
    ]
    monkeypatch.setattr(
        "osa_anurans_app.src.candidates.iter_uncertainty_scores", lambda *args: iter(scores)
    )
    pools = _uncertainty_pools(
        [object()],
        object(),
        4,
        1,
        excluded_windows={"A": {f"known-{i}" for i in range(20)}},
    )
    assert pools["A"] == [("fresh-a", 0.1), ("fresh-b", 0.2)]
    assert pools["B"] == [("known-0", 0)]
