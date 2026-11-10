"""One-time seed rows preserve species presence and stable retry identities."""

import pytest
from osa_anurans_app.scripts.seed_grillos import seed_rows
from osa_anurans_app.src.records import LabelCounts


def test_seed_preserves_multiple_species_and_stable_ids():
    labels = [LabelCounts("window", "A", present=1), LabelCounts("window", "B", present=1)]
    first = seed_rows(labels, "2026-11-07T00:00:00+00:00")
    retry = seed_rows(list(reversed(labels)), "2026-11-08T00:00:00+00:00")
    assert {r["annotation_id"] for r in first} == {r["annotation_id"] for r in retry}
    assert len({r["annotation_id"] for r in first}) == 2
    assert {r["query_species_id"] for r in first} == {"A", "B"}
    assert all(r["annotator_id"] == r["annotator_name"] == "grillos-seed-gt" for r in first)
    assert all(r["candidate_id"] is None and r["response"] == "present" for r in first)


@pytest.mark.parametrize(
    "labels",
    [
        [],
        [LabelCounts("w", "A", absent=1)],
        [LabelCounts("w", "A", present=1, absent=1)],
        [LabelCounts("w", "A", present=2)],
        [LabelCounts("w", "A", present=1)] * 2,
    ],
)
def test_seed_rejects_non_bootstrap_or_duplicate_votes(labels):
    with pytest.raises(ValueError):
        seed_rows(labels, "2026-11-07T00:00:00+00:00")
