"""Check per-species selection against exhaustive scoring, including tied scores."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from osa_anurans_app.src.candidates import iter_candidates, write_candidates
from osa_anurans_app.src.config import AppConfig, load_config, validate_config
from osa_anurans_app.src.model.model import LinearProbe, ProbeArtifact
from osa_anurans_app.src.records import EmbeddingRow


@pytest.mark.parametrize("limit", [2, 100])
def test_saved_candidates_keep_lowest_scores_per_species_with_stable_ties(tmp_path, limit):
    path = tmp_path / "embeddings.npy"
    np.save(path, np.array([[0, 0], [1, 0], [-1, 0], [0, 0]], dtype=np.float32))
    rows = [EmbeddingRow(f"w{i}", "recording", "config", path, i) for i in range(4)]
    model = LinearProbe(2, 2)
    with torch.no_grad():
        model.probe.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        model.probe.bias.zero_()
    artifact = ProbeArtifact(model, ("A", "B"), "config", "run")
    exhaustive = list(iter_candidates(rows, artifact))
    output = tmp_path / "candidates.jsonl"
    # Reversing inputs must not change which tied windows are selected.
    count = write_candidates(output, list(reversed(rows)), artifact, 3, limit)
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == len(saved) == 2 * min(limit, len(rows))
    for species in artifact.species_ids:
        expected = sorted(
            [row for row in exhaustive if row["query_species_id"] == species],
            key=lambda row: (row["score"], row["window_id"]),
        )[:limit]
        actual = [row for row in saved if row["query_species_id"] == species]
        assert [(row["candidate_id"], row["score"]) for row in actual] == [
            (row["candidate_id"], row["score"]) for row in expected
        ]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_candidate_limit_is_rejected(tmp_path, limit):
    config = AppConfig((tmp_path,), tmp_path / "annotations.csv", tmp_path)
    assert config.candidates_per_species == 100
    with pytest.raises(ValueError, match="candidates_per_species"):
        validate_config(replace(config, candidates_per_species=limit))
    with pytest.raises(ValueError, match="candidates_per_species"):
        write_candidates(tmp_path / "candidates.jsonl", [], None, 4, limit)


def test_candidate_limit_loads_from_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[paths]\naudio_roots = ['.']\nannotations_csv = 'annotations.csv'\n"
        "[candidates]\ncandidates_per_species = 3\n",
        encoding="utf-8",
    )
    assert load_config(config_path).candidates_per_species == 3
