#!/usr/bin/env python3
"""
Build research-grade multi-label targets for the AnuraSet 3-second dataset.

Inputs
------
metadata.csv
recording_weak_labels.csv

The script expects metadata.csv to contain, at minimum:
    segment_id, recording_id, site_id, split, source_type

For strong_event rows it uses:
    event_label

If present, it also uses:
    labels
    event_start, event_end

Outputs
-------
species_vocabulary.csv
strong_targets.csv
recording_weak_labels_long.csv
segment_targets.csv
target_summary.csv

Important semantics
-------------------
Strong labels:
    1    confirmed species present in the localized strong annotation
    NaN  unknown / not established as absent

Weak recording labels:
    0    absence at the 1-minute recording level
    1    Low activity
    2    Moderate activity
    3    High activity

Weak recording-level values are NEVER copied to individual 3-second
segments as segment-level targets.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_METADATA = {
    "segment_id",
    "recording_id",
    "site_id",
    "split",
    "source_type",
}

QUALITY_RE = re.compile(r"^(?P<species>.+)_(?P<quality>[LMH])$", re.IGNORECASE)


def normalize_species(label) -> str | None:
    """Normalize a strong annotation label to its biological species code."""
    if pd.isna(label):
        return None

    s = str(label).strip().upper()
    if not s:
        return None

    # Strong annotations may use ADEMAR_L/M/H where the suffix is
    # recording-quality metadata rather than a biological class.
    m = QUALITY_RE.match(s)
    if m:
        s = m.group("species")

    # Handle accidental whitespace around separators.
    s = s.strip()
    return s or None


def split_label_string(value) -> list[str]:
    """Parse common representations of multi-label strings."""
    if pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    # Python-list representation, e.g. "['ADEMAR', 'PHYDIS']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed]
        except Exception:
            pass

    # Common delimiters used in metadata.
    parts = re.split(r"[|;,]+", s)
    return [p.strip() for p in parts if p.strip()]


def extract_strong_species(row) -> list[str]:
    """
    Extract the COMPLETE merged strong-species set for a segment.

    The segment builder explicitly merges every strong annotation that
    overlaps the same 3-second window. Therefore the canonical source is:

        strong_species_labels

    Older metadata may not contain that field, so we retain a backwards-
    compatible fallback to labels/event_label. The fallback should not be
    used when strong_species_labels is present.
    """
    candidates = []

    # Canonical field produced by the corrected multi-label builder.
    if "strong_species_labels" in row.index:
        value = row.get("strong_species_labels", np.nan)
        if not pd.isna(value):
            candidates.extend(split_label_string(value))

    # Backwards-compatible fields for older datasets.
    if not candidates and "labels" in row.index:
        value = row.get("labels", np.nan)
        if not pd.isna(value):
            candidates.extend(split_label_string(value))

    if not candidates and "event_label" in row.index:
        value = row.get("event_label", np.nan)
        if not pd.isna(value):
            candidates.extend(split_label_string(value))

    normalized = []
    for label in candidates:
        sp = normalize_species(label)
        if sp and sp not in normalized:
            normalized.append(sp)

    return sorted(normalized)


def validate_metadata(df: pd.DataFrame):
    missing = REQUIRED_METADATA - set(df.columns)
    if missing:
        raise ValueError(
            "metadata.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df["segment_id"].duplicated().any():
        dup = int(df["segment_id"].duplicated().sum())
        raise ValueError(f"segment_id is not unique; duplicate rows: {dup}")

    if df["recording_id"].isna().any():
        raise ValueError("Some segments have missing recording_id.")

    unknown_sites = sorted(
        df.loc[df["site_id"].isna(), "segment_id"].astype(str).unique()
    )
    if unknown_sites:
        raise ValueError(
            f"Found {len(unknown_sites)} segments with missing site_id."
        )

    if (df["site_id"].astype(str).str.upper() == "UNKNOWN").any():
        n = int(
            (df["site_id"].astype(str).str.upper() == "UNKNOWN").sum()
        )
        raise ValueError(f"Found {n} segments with site_id='UNKNOWN'.")


def load_weak_labels(path: Path) -> pd.DataFrame:
    weak = pd.read_csv(path)

    required = {"recording_id"}
    missing = required - set(weak.columns)
    if missing:
        raise ValueError(
            "recording_weak_labels.csv is missing: "
            + ", ".join(sorted(missing))
        )

    # If a separate site_id exists, keep it for consistency checks.
    return weak


def clean_weak_labels(weak: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean recording-level weak labels before expanding them to long format.

    The weak-label CSV is recording-level.  A row is retained only when its
    recording_id occurs in metadata, recording IDs are normalized, and duplicate
    recording IDs are resolved deterministically. Species columns are preserved.
    This prevents out-of-dataset recordings from inflating the weak-label table.
    """
    weak = weak.copy()
    meta_ids = metadata["recording_id"].astype(str).str.strip()

    weak["recording_id"] = weak["recording_id"].astype(str).str.strip()
    weak = weak[weak["recording_id"].ne("") & weak["recording_id"].ne("nan")].copy()

    before_rows = len(weak)
    before_recordings = weak["recording_id"].nunique()

    duplicate_recordings = int(weak["recording_id"].duplicated(keep=False).sum())
    if duplicate_recordings:
        # Fail rather than silently combining conflicting recording-level labels.
        dup_ids = weak.loc[weak["recording_id"].duplicated(keep=False), "recording_id"]
        raise ValueError(
            "recording_weak_labels.csv contains duplicate recording_id rows. "
            f"Found {dup_ids.nunique()} duplicated recording IDs; resolve them "
            "before target generation."
        )

    meta_id_set = set(meta_ids)
    weak_id_set = set(weak["recording_id"])
    missing_from_metadata = sorted(weak_id_set - meta_id_set)
    weak = weak[weak["recording_id"].isin(meta_id_set)].copy()

    stats = {
        "weak_rows_before_cleanup": before_rows,
        "weak_recordings_before_cleanup": before_recordings,
        "weak_rows_after_cleanup": len(weak),
        "weak_recordings_after_cleanup": weak["recording_id"].nunique(),
        "weak_recordings_removed_not_in_metadata": len(missing_from_metadata),
        "weak_recording_ids_removed_not_in_metadata": "|".join(missing_from_metadata),
    }

    if missing_from_metadata:
        print(
            f"CLEANUP: removed {len(missing_from_metadata):,} weak recordings "
            "not present in metadata."
        )

    return weak, stats


def build_weak_long(weak: pd.DataFrame) -> pd.DataFrame:
    species_cols = [c for c in weak.columns if c.startswith("SPECIES_")]

    records = []
    for _, row in weak.iterrows():
        recording_id = row["recording_id"]

        for col in species_cols:
            value = row[col]

            if pd.isna(value):
                continue

            try:
                activity = int(value)
            except Exception as exc:
                raise ValueError(
                    f"Non-integer weak activity in {col} for "
                    f"{recording_id}: {value!r}"
                ) from exc

            if activity not in (0, 1, 2, 3):
                raise ValueError(
                    f"Weak activity must be 0/1/2/3; found {activity} "
                    f"in {col} for {recording_id}"
                )

            species = col.removeprefix("SPECIES_").upper()

            records.append(
                {
                    "recording_id": recording_id,
                    "species": species,
                    "activity": activity,
                    "presence": int(activity > 0),
                }
            )

    return pd.DataFrame(
        records,
        columns=["recording_id", "species", "activity", "presence"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing metadata.csv and recording_weak_labels.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to dataset-dir/targets",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir or (dataset_dir / "targets")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / "metadata.csv"
    weak_path = dataset_dir / "recording_weak_labels.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not weak_path.exists():
        raise FileNotFoundError(weak_path)

    print(f"Reading metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    validate_metadata(meta)

    print(f"Segments: {len(meta):,}")
    print(f"Recordings: {meta['recording_id'].nunique():,}")
    print(f"Sites: {meta['site_id'].nunique():,}")

    # ------------------------------------------------------------
    # Strong targets
    # ------------------------------------------------------------
    # The corrected segment builder writes the complete merged species set.
    # If this field exists, it is the authoritative strong multi-label source.
    has_merged_builder_field = "strong_species_labels" in meta.columns

    if has_merged_builder_field:
        print("Using builder field: strong_species_labels")
    else:
        print(
            "WARNING: strong_species_labels is absent. "
            "Using legacy labels/event_label fallback."
        )

    meta["_strong_species"] = meta.apply(
        extract_strong_species, axis=1
    )

    # Only strong-event rows can establish temporally localized
    # strong presence.
    strong_mask = meta["source_type"].eq("strong_event")

    strong_rows = []
    for _, row in meta.loc[strong_mask].iterrows():
        species = row["_strong_species"]

        strong_rows.append(
            {
                "segment_id": row["segment_id"],
                "recording_id": row["recording_id"],
                "site_id": row["site_id"],
                "split": row["split"],
                "source_type": row["source_type"],
                "strong_labels": "|".join(species),
                "n_strong_species": len(species),
            }
        )

    strong_info = pd.DataFrame(strong_rows)

    # Species discovered in strong annotations.
    strong_species = set()
    for labels in meta.loc[strong_mask, "_strong_species"]:
        strong_species.update(labels)

    # ------------------------------------------------------------
    # Weak labels
    # ------------------------------------------------------------
    print(f"Reading weak labels: {weak_path}")
    weak = load_weak_labels(weak_path)

    # ------------------------------------------------------------
    # Cleanup weak recording table BEFORE long-format expansion
    # ------------------------------------------------------------
    weak, weak_cleanup = clean_weak_labels(weak, meta)

    weak_long = build_weak_long(weak)

    if not weak_long.empty:
        weak_long.to_csv(
            out_dir / "recording_weak_labels_long.csv",
            index=False,
        )
    else:
        weak_long.to_csv(
            out_dir / "recording_weak_labels_long.csv",
            index=False,
        )

    weak_species = set(weak_long["species"].unique()) if not weak_long.empty else set()

    # ------------------------------------------------------------
    # Species vocabulary
    # ------------------------------------------------------------
    all_species = sorted(str(x).upper() for x in (strong_species | weak_species))

    vocab = pd.DataFrame(
        {
            "species_index": np.arange(len(all_species), dtype=int),
            "species_code": all_species,
        }
    )
    vocab.to_csv(out_dir / "species_vocabulary.csv", index=False)

    # ------------------------------------------------------------
    # Strong multi-hot / partial-label target matrix
    # ------------------------------------------------------------
    # NaN means unknown. 1 means confirmed present.
    #
    # We deliberately do not turn every non-annotated species into 0.
    strong_target = pd.DataFrame(
        np.nan,
        index=meta.index,
        columns=all_species,
    )

    # Weak candidates have no segment-level strong target information.
    # Strong-event rows receive 1 for every species supported by their
    # localized annotation.
    for idx, species_list in meta.loc[strong_mask, "_strong_species"].items():
        for species in species_list:
            strong_target.at[idx, species] = 1.0

    strong_target_out = pd.concat(
        [
            meta[
                [
                    "segment_id",
                    "recording_id",
                    "site_id",
                    "split",
                    "source_type",
                ]
            ].reset_index(drop=True),
            strong_target.reset_index(drop=True),
        ],
        axis=1,
    )

    strong_target_out.to_csv(
        out_dir / "strong_targets.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Master segment target metadata
    # ------------------------------------------------------------
    master = meta[
        [
            "segment_id",
            "recording_id",
            "site_id",
            "recording_date",
            "recording_time",
            "split",
            "source_type",
        ]
    ].copy()

    master["strong_labels"] = meta["_strong_species"].apply(
        lambda x: "|".join(x)
    )
    master["n_strong_species"] = meta["_strong_species"].apply(len)

    # Attach recording-level weak labels as a compact audit field.
    weak_by_recording = {}
    if not weak_long.empty:
        positive = weak_long[weak_long["activity"] > 0]
        for recording_id, group in positive.groupby("recording_id"):
            pairs = [
                f"{row.species}:{int(row.activity)}"
                for row in group.itertuples(index=False)
            ]
            weak_by_recording[recording_id] = "|".join(pairs)

    master["weak_recording_labels"] = master["recording_id"].map(
        weak_by_recording
    )

    presence_by_recording = {}
    if not weak_long.empty:
        positive = weak_long[weak_long["presence"] == 1]
        for recording_id, group in positive.groupby("recording_id"):
            presence_by_recording[recording_id] = "|".join(
                sorted(group["species"].unique())
            )

    master["weak_species_present"] = master["recording_id"].map(
        presence_by_recording
    )

    master.to_csv(out_dir / "segment_targets.csv", index=False)

    # ------------------------------------------------------------
    # Quality / sanity summary
    # ------------------------------------------------------------
    n_multi = int(
        ((meta["source_type"] == "strong_event")
         & (meta["n_strong_species"] > 1)).sum()
    )

    n_zero_strong = int(
        ((meta["source_type"] == "strong_event")
         & (meta["n_strong_species"] == 0)).sum()
    )
    if n_zero_strong:
        raise ValueError(
            f"{n_zero_strong} strong-event segments have no extracted "
            "species labels."
        )

    strong_species_counts = {}
    for species in all_species:
        strong_species_counts[species] = int(
            strong_target[species].eq(1).sum()
        )

    summary_rows = [
        ("n_segments", len(meta)),
        ("n_recordings", meta["recording_id"].nunique()),
        ("n_sites", meta["site_id"].nunique()),
        ("n_species", len(all_species)),
        ("n_strong_segments", int(strong_mask.sum())),
        (
            "n_weak_candidate_segments",
            int((~strong_mask).sum()),
        ),
        ("n_multi_species_strong_segments", n_multi),
        ("used_merged_builder_labels", int(has_merged_builder_field)),
        ("n_weak_recordings", weak["recording_id"].nunique()),
        (
            "n_weak_recordings_removed_not_in_metadata",
            weak_cleanup["weak_recordings_removed_not_in_metadata"],
        ),
        (
            "n_metadata_recordings_missing_from_weak_table",
            len(
                set(meta["recording_id"].astype(str))
                - set(weak["recording_id"].astype(str))
            ),
        ),
    ]

    summary_rows.extend([
        ("weak_rows_before_cleanup", weak_cleanup["weak_rows_before_cleanup"]),
        ("weak_recordings_before_cleanup", weak_cleanup["weak_recordings_before_cleanup"]),
        ("weak_rows_after_cleanup", weak_cleanup["weak_rows_after_cleanup"]),
        ("weak_recordings_after_cleanup", weak_cleanup["weak_recordings_after_cleanup"]),
    ])

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary.to_csv(out_dir / "target_summary.csv", index=False)

    counts = pd.DataFrame(
        {
            "species_code": all_species,
            "strong_positive_segments": [
                strong_species_counts[s] for s in all_species
            ],
            "weak_recording_positive_count": [
                int(
                    (
                        weak_long.loc[
                            weak_long["species"].eq(s), "presence"
                        ] == 1
                    ).sum()
                )
                if not weak_long.empty
                else 0
                for s in all_species
            ],
        }
    )
    counts.to_csv(out_dir / "species_target_counts.csv", index=False)

    # Save the cleaned recording-level source table for auditability.
    weak.to_csv(out_dir / "recording_weak_labels_clean.csv", index=False)

    # ------------------------------------------------------------
    # Hard checks
    # ------------------------------------------------------------
    assert len(vocab) == len(all_species)
    assert strong_target_out["segment_id"].is_unique
    assert master["segment_id"].is_unique

    if not weak_long.empty:
        assert weak_long["activity"].isin([0, 1, 2, 3]).all()

    if "site_id" in weak.columns and "site_id" in meta.columns:
        weak_site = (
            weak[["recording_id", "site_id"]]
            .dropna(subset=["recording_id"])
            .drop_duplicates()
        )
        meta_rec_site = (
            meta[["recording_id", "site_id"]]
            .drop_duplicates()
        )
        joined = weak_site.merge(
            meta_rec_site,
            on="recording_id",
            how="inner",
            suffixes=("_weak", "_meta"),
        )
        mismatches = joined[
            joined["site_id_weak"].astype(str)
            != joined["site_id_meta"].astype(str)
        ]
        if len(mismatches):
            print(
                f"WARNING: {len(mismatches)} recording/site mismatches "
                "between weak labels and metadata."
            )

    print("\nDONE")
    print(f"Output directory: {out_dir}")
    print(f"Species vocabulary: {len(all_species)} species")
    print(f"Strong segments: {strong_mask.sum():,}")
    print(f"Multi-species strong segments: {n_multi:,}")
    print(
        f"Strong segments with >=2 species: "
        f"{n_multi:,} ({100*n_multi/max(int(strong_mask.sum()),1):.1f}%)"
    )
    print(f"Weak recordings: {weak['recording_id'].nunique():,}")
    print("\nCreated:")
    for name in [
        "species_vocabulary.csv",
        "strong_targets.csv",
        "recording_weak_labels_long.csv",
        "segment_targets.csv",
        "target_summary.csv",
        "species_target_counts.csv",
        "recording_weak_labels_clean.csv",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
