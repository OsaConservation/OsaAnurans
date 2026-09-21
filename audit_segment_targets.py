#!/usr/bin/env python3
"""
Research-grade audit for RANA segment_targets.csv.

Default dataset:
D:\Acoustics\AnuraSet_3sec_all\

Expected inputs:
    metadata.csv
    segment_targets.csv
    birdnet_embeddings\
        embeddings.npy
        embedding_metadata.csv

The audit checks:
1. Segment-target row count and segment_id integrity.
2. Exact segment_id set/order alignment with metadata and embeddings.
3. Duplicate/missing IDs.
4. Required metadata columns and source_type consistency.
5. Target column discovery and binary/NaN validity.
6. Strong-label semantics: 1 = confirmed presence; NaN = unknown.
7. Weak-label semantics: if source columns are present, values must be
   consistent with activity/presence conventions.
8. No impossible values in model-ready target columns.
9. Multilabel structure: number of positive labels per segment.
10. Target prevalence by species.
11. Strong/weak source distributions.
12. Site distribution of target rows.
13. Cross-check against embeddings.npy and embedding_metadata.csv.
14. Optional strong_targets.csv, species_vocabulary.csv,
    species_target_counts.csv, target_summary.csv, and
    recording_weak_labels_clean.csv if present.

IMPORTANT:
- This script is read-only.
- It does not change target files.
- It does not infer missing labels as absences.
- It treats NaN as unknown unless a binary target column explicitly contains
  a documented absence value.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(r"D:\Acoustics\AnuraSet_3sec_all")
EXPECTED_EMBED_DIM = 1024


def log(msg=""):
    print(msg, flush=True)


def normalize_ids(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def existing_col(df: pd.DataFrame, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def is_species_column(name: str) -> bool:
    return (
        name.startswith("SPECIES_")
        or name.startswith("species_")
        or name.startswith("target_")
    )


def safe_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def check(condition, passed, failed, message):
    if condition:
        log(f"[PASS] {message}")
        return passed + 1, failed
    log(f"[FAIL] {message}")
    return passed, failed + 1


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def audit(args):
    root = Path(args.dataset_dir)
    targets_path = Path(args.targets) if args.targets else root / "segment_targets.csv"
    metadata_path = Path(args.metadata) if args.metadata else root / "metadata.csv"
    emb_dir = Path(args.embedding_dir) if args.embedding_dir else root / "birdnet_embeddings"
    embeddings_path = emb_dir / "embeddings.npy"
    emb_meta_path = emb_dir / "embedding_metadata.csv"

    optional_paths = {
        "strong_targets": root / "strong_targets.csv",
        "species_vocabulary": root / "species_vocabulary.csv",
        "species_target_counts": root / "species_target_counts.csv",
        "target_summary": root / "target_summary.csv",
        "recording_weak_labels_clean": root / "recording_weak_labels_clean.csv",
    }

    report_dir = Path(args.report_dir) if args.report_dir else root / "target_audit"
    report_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"segment_target_audit_{stamp}.json"
    txt_path = report_dir / f"segment_target_audit_{stamp}.txt"

    passed = 0
    failed = 0
    warnings = []
    errors = []
    findings = {}

    log("=" * 78)
    log("RANA / SEGMENT TARGET DATASET AUDIT")
    log("=" * 78)
    log(f"Dataset root: {root}")
    log(f"Targets:      {targets_path}")
    log(f"Metadata:     {metadata_path}")
    log(f"Embeddings:   {embeddings_path}")
    log(f"Emb. meta:    {emb_meta_path}")
    log("")

    # ------------------------------------------------------------------
    # Required files
    # ------------------------------------------------------------------
    required = {
        "segment_targets": targets_path,
        "metadata": metadata_path,
        "embeddings": embeddings_path,
        "embedding_metadata": emb_meta_path,
    }

    for key, path in required.items():
        if path.exists():
            log(f"[PASS] {key}: {path}")
            passed += 1
        else:
            log(f"[FAIL] {key} not found: {path}")
            failed += 1
            errors.append(f"Required file not found: {path}")

    for key, path in optional_paths.items():
        if path.exists():
            log(f"[INFO] Optional file present: {key}")
        else:
            warnings.append(f"Optional file not present: {path}")

    if failed:
        log("")
        log("Cannot continue because a required file is missing.")
        findings["status"] = "FAIL"
        findings["summary"] = {
            "passed_checks": passed,
            "failed_checks": failed,
            "warnings": len(warnings),
        }
        json_path.write_text(
            json.dumps(findings, indent=2, default=safe_json),
            encoding="utf-8",
        )
        txt_path.write_text(
            "RANA segment target audit\n\n" + "\n".join(errors),
            encoding="utf-8",
        )
        log(f"JSON report: {json_path}")
        log(f"TXT report:  {txt_path}")
        return 2

    # ------------------------------------------------------------------
    # Load target and metadata files
    # ------------------------------------------------------------------
    log("")
    log("Loading segment_targets.csv...")
    targets = read_csv(targets_path)

    log("Loading metadata.csv...")
    metadata = read_csv(metadata_path)

    findings["targets"] = {
        "rows": int(len(targets)),
        "columns": list(targets.columns),
    }
    findings["metadata"] = {
        "rows": int(len(metadata)),
        "columns": list(metadata.columns),
    }

    target_id_col = existing_col(targets, ["segment_id", "segment", "id"])
    metadata_id_col = existing_col(metadata, ["segment_id", "segment", "id"])

    passed, failed = check(
        target_id_col is not None,
        passed, failed,
        f"segment_targets contains segment_id (found {target_id_col})",
    )
    passed, failed = check(
        metadata_id_col is not None,
        passed, failed,
        f"metadata.csv contains segment_id (found {metadata_id_col})",
    )

    if target_id_col is None or metadata_id_col is None:
        errors.append("Cannot perform target alignment without segment_id.")
    else:
        target_ids = normalize_ids(targets[target_id_col])
        metadata_ids = normalize_ids(metadata[metadata_id_col])

        target_missing = int(target_ids.isna().sum())
        target_dupes = int(target_ids.dropna().duplicated().sum())
        metadata_missing = int(metadata_ids.isna().sum())
        metadata_dupes = int(metadata_ids.dropna().duplicated().sum())

        passed, failed = check(
            target_missing == 0,
            passed, failed,
            f"No missing target segment_id values (found {target_missing})",
        )
        passed, failed = check(
            target_dupes == 0,
            passed, failed,
            f"No duplicate target segment_id values (found {target_dupes})",
        )
        passed, failed = check(
            metadata_missing == 0,
            passed, failed,
            f"No missing metadata segment_id values (found {metadata_missing})",
        )
        passed, failed = check(
            metadata_dupes == 0,
            passed, failed,
            f"No duplicate metadata segment_id values (found {metadata_dupes})",
        )

        findings["id_integrity"] = {
            "target_missing_ids": target_missing,
            "target_duplicate_ids": target_dupes,
            "metadata_missing_ids": metadata_missing,
            "metadata_duplicate_ids": metadata_dupes,
        }

        # Exact row count and exact order.
        passed, failed = check(
            len(targets) == len(metadata),
            passed, failed,
            f"Target rows equal metadata rows ({len(targets)} == {len(metadata)})",
        )

        if len(targets) == len(metadata):
            exact_order = target_ids.fillna("<NA>").reset_index(drop=True).equals(
                metadata_ids.fillna("<NA>").reset_index(drop=True)
            )
            passed, failed = check(
                exact_order,
                passed, failed,
                "segment_targets segment_id order exactly matches metadata.csv",
            )

        target_set = set(target_ids.dropna())
        metadata_set = set(metadata_ids.dropna())

        only_targets = sorted(target_set - metadata_set)
        only_metadata = sorted(metadata_set - target_set)

        passed, failed = check(
            not only_targets,
            passed, failed,
            f"No target segment_ids absent from metadata (extra={len(only_targets)})",
        )
        passed, failed = check(
            not only_metadata,
            passed, failed,
            f"No metadata segment_ids absent from targets (missing={len(only_metadata)})",
        )

        findings["id_alignment"] = {
            "exact_order": (
                bool(exact_order) if len(targets) == len(metadata) else False
            ),
            "target_only_count": len(only_targets),
            "metadata_only_count": len(only_metadata),
            "target_only_examples": only_targets[:20],
            "metadata_only_examples": only_metadata[:20],
        }

    # ------------------------------------------------------------------
    # Metadata columns
    # ------------------------------------------------------------------
    log("")
    log("Checking metadata structure...")

    required_metadata_cols = [
        "recording_id",
        "site_id",
        "split",
        "source_type",
    ]

    for col in required_metadata_cols:
        if col in metadata.columns:
            passed += 1
            log(f"[PASS] metadata contains {col}")
        else:
            failed += 1
            log(f"[FAIL] metadata missing {col}")
            errors.append(f"metadata.csv missing required column: {col}")

    for col in required_metadata_cols:
        if col in targets.columns:
            log(f"[INFO] targets also contains {col}")

    # Compare metadata fields when target file also carries them.
    for col in ["recording_id", "site_id", "split", "source_type"]:
        if col in metadata.columns and col in targets.columns:
            a = normalize_ids(metadata[col])
            b = normalize_ids(targets[col])
            if len(a) == len(b):
                same = a.fillna("<NA>").reset_index(drop=True).equals(
                    b.fillna("<NA>").reset_index(drop=True)
                )
                passed, failed = check(
                    same,
                    passed, failed,
                    f"targets {col} exactly matches metadata.csv",
                )

    if "site_id" in metadata.columns:
        site_counts = metadata["site_id"].astype("string").fillna("<MISSING>").value_counts()
        findings["site_distribution"] = {str(k): int(v) for k, v in site_counts.items()}
        log("")
        log("Site distribution:")
        for site, n in site_counts.items():
            log(f"  {site}: {n}")

    if "source_type" in metadata.columns:
        source_counts = metadata["source_type"].astype("string").fillna("<MISSING>").value_counts()
        findings["source_type_distribution"] = {
            str(k): int(v) for k, v in source_counts.items()
        }
        log("")
        log("Source type distribution:")
        for source, n in source_counts.items():
            log(f"  {source}: {n}")

    # ------------------------------------------------------------------
    # Target column discovery
    # ------------------------------------------------------------------
    id_and_metadata = {
        "segment_id", "recording_id", "site_id", "split", "source_type",
        "labels", "strong_species_labels", "event_label",
        "species_labels", "target_source", "label_source",
    }

    target_cols = [
        c for c in targets.columns
        if c not in id_and_metadata and is_species_column(c)
    ]

    # If no SPECIES_/target_ columns, inspect numeric/bool columns as a fallback.
    if not target_cols:
        fallback = []
        for c in targets.columns:
            if c in id_and_metadata:
                continue
            if pd.api.types.is_numeric_dtype(targets[c]) or pd.api.types.is_bool_dtype(targets[c]):
                fallback.append(c)
        target_cols = fallback

    findings["target_columns"] = {
        "count": len(target_cols),
        "columns": target_cols,
    }

    passed, failed = check(
        len(target_cols) > 0,
        passed, failed,
        f"Found {len(target_cols)} model target columns",
    )

    log("")
    log(f"Target columns detected: {len(target_cols)}")
    if target_cols:
        log("  " + ", ".join(target_cols))

    # ------------------------------------------------------------------
    # Target value validation
    # ------------------------------------------------------------------
    positive_counts = {}
    negative_counts = {}
    unknown_counts = {}
    invalid_values = {}
    target_density = []

    for col in target_cols:
        s = targets[col]

        # Convert common string representations to numeric where possible.
        numeric = pd.to_numeric(s, errors="coerce")

        # Presence/absence target semantics:
        # valid = NaN or 0 or 1.
        invalid_mask = s.notna() & numeric.isna()
        invalid_numeric = numeric.notna() & ~numeric.isin([0, 1])

        invalid_count = int((invalid_mask | invalid_numeric).sum())

        pos = int((numeric == 1).sum())
        neg = int((numeric == 0).sum())
        unk = int(numeric.isna().sum())

        positive_counts[col] = pos
        negative_counts[col] = neg
        unknown_counts[col] = unk
        invalid_values[col] = invalid_count

        target_density.append(pos)

        passed, failed = check(
            invalid_count == 0,
            passed, failed,
            f"{col}: target values are only 0/1/NaN (invalid={invalid_count})",
        )

    findings["target_statistics"] = {
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "unknown_counts": unknown_counts,
        "invalid_counts": invalid_values,
    }

    # ------------------------------------------------------------------
    # Multilabel structure
    # ------------------------------------------------------------------
    if target_cols:
        numeric_targets = targets[target_cols].apply(pd.to_numeric, errors="coerce")
        positives_per_segment = numeric_targets.eq(1).sum(axis=1)
        known_per_segment = numeric_targets.notna().sum(axis=1)

        findings["multilabel"] = {
            "segments_with_zero_positive_labels": int((positives_per_segment == 0).sum()),
            "segments_with_one_positive_label": int((positives_per_segment == 1).sum()),
            "segments_with_multiple_positive_labels": int((positives_per_segment > 1).sum()),
            "maximum_positive_labels_in_segment": int(positives_per_segment.max()),
            "mean_positive_labels_per_segment": float(positives_per_segment.mean()),
            "segments_with_all_targets_unknown": int((known_per_segment == 0).sum()),
            "segments_with_at_least_one_known_target": int((known_per_segment > 0).sum()),
        }

        log("")
        log("Multilabel structure:")
        log(f"  0 positive species:       {(positives_per_segment == 0).sum()}")
        log(f"  1 positive species:       {(positives_per_segment == 1).sum()}")
        log(f"  >1 positive species:      {(positives_per_segment > 1).sum()}")
        log(f"  Maximum positives/segment:{positives_per_segment.max()}")
        log(f"  Mean positives/segment:   {positives_per_segment.mean():.4f}")
        log(f"  All targets unknown:      {(known_per_segment == 0).sum()}")

        # A multilabel dataset should have at least some overlapping positives.
        if int((positives_per_segment > 1).sum()) == 0:
            warnings.append(
                "No segment has more than one positive species. "
                "This may be valid, but it means overlap is absent from the current targets."
            )

        # Warn, don't fail, if all targets are known binary values.
        if int((known_per_segment == 0).sum()) > 0:
            warnings.append(
                f"{int((known_per_segment == 0).sum())} segments have all target columns unknown."
            )

    # ------------------------------------------------------------------
    # Strong/weak semantics
    # ------------------------------------------------------------------
    if "source_type" in metadata.columns:
        source = metadata["source_type"].astype("string")
        known_sources = set(source.dropna().unique())

        findings["source_type_values"] = sorted(map(str, known_sources))

        # For strong_event rows, if target columns exist, presence values are valid
        # but absence must not be inferred merely because a species is NaN.
        if "strong_event" in known_sources and target_cols:
            strong_mask = source.eq("strong_event")
            strong_df = targets.loc[strong_mask, target_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            strong_positive = int(strong_df.eq(1).sum().sum())
            strong_unknown = int(strong_df.isna().sum().sum())

            findings["strong_targets"] = {
                "rows": int(strong_mask.sum()),
                "positive_cells": strong_positive,
                "unknown_cells": strong_unknown,
            }

            log("")
            log("Strong-event target semantics:")
            log(f"  Strong-event rows: {strong_mask.sum()}")
            log(f"  Positive target cells: {strong_positive}")
            log(f"  Unknown target cells: {strong_unknown}")

        # Weak candidate rows should not automatically be all positive.
        if "weak_candidate" in known_sources and target_cols:
            weak_mask = source.eq("weak_candidate")
            weak_df = targets.loc[weak_mask, target_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            weak_positive = int(weak_df.eq(1).sum().sum())
            weak_zero = int(weak_df.eq(0).sum().sum())
            weak_unknown = int(weak_df.isna().sum().sum())

            findings["weak_targets"] = {
                "rows": int(weak_mask.sum()),
                "positive_cells": weak_positive,
                "zero_cells": weak_zero,
                "unknown_cells": weak_unknown,
            }

            log("")
            log("Weak-candidate target semantics:")
            log(f"  Weak-candidate rows: {weak_mask.sum()}")
            log(f"  Positive target cells: {weak_positive}")
            log(f"  Zero target cells:      {weak_zero}")
            log(f"  Unknown target cells:   {weak_unknown}")

    # ------------------------------------------------------------------
    # Species prevalence
    # ------------------------------------------------------------------
    if target_cols:
        rows = []
        for col in target_cols:
            pos = positive_counts[col]
            zero = negative_counts[col]
            unk = unknown_counts[col]
            known = pos + zero
            prevalence = pos / known if known else float("nan")
            rows.append({
                "species": col,
                "positive_segments": pos,
                "negative_segments": zero,
                "unknown_segments": unk,
                "known_segments": known,
                "prevalence_among_known": prevalence,
            })

        prevalence_df = pd.DataFrame(rows).sort_values(
            ["positive_segments", "species"],
            ascending=[False, True],
        )
        findings["species_prevalence"] = rows

        log("")
        log("Species target prevalence:")
        for _, r in prevalence_df.iterrows():
            prev = (
                f"{r['prevalence_among_known']:.4f}"
                if math.isfinite(r["prevalence_among_known"])
                else "NA"
            )
            log(
                f"  {r['species']}: +{int(r['positive_segments'])}, "
                f"0={int(r['negative_segments'])}, "
                f"unknown={int(r['unknown_segments'])}, "
                f"prevalence={prev}"
            )

    # ------------------------------------------------------------------
    # Optional supporting files
    # ------------------------------------------------------------------
    if optional_paths["species_vocabulary"].exists():
        try:
            vocab = read_csv(optional_paths["species_vocabulary"])
            findings["species_vocabulary"] = {
                "rows": int(len(vocab)),
                "columns": list(vocab.columns),
            }

            # Best-effort species column detection.
            vocab_col = existing_col(vocab, ["species", "species_name", "target", "label"])
            if vocab_col and target_cols:
                vocab_names = set(vocab[vocab_col].astype("string").dropna())
                target_names = set(target_cols)
                findings["species_vocabulary"]["missing_in_targets"] = sorted(
                    vocab_names - target_names
                )
                findings["species_vocabulary"]["missing_in_vocabulary"] = sorted(
                    target_names - vocab_names
                )
        except Exception as exc:
            warnings.append(f"Could not audit species_vocabulary.csv: {exc}")

    if optional_paths["species_target_counts"].exists():
        try:
            counts = read_csv(optional_paths["species_target_counts"])
            findings["species_target_counts_file"] = {
                "rows": int(len(counts)),
                "columns": list(counts.columns),
            }
        except Exception as exc:
            warnings.append(f"Could not audit species_target_counts.csv: {exc}")

    if optional_paths["strong_targets"].exists():
        try:
            strong = read_csv(optional_paths["strong_targets"])
            findings["strong_targets_file"] = {
                "rows": int(len(strong)),
                "columns": list(strong.columns),
            }
            if target_id_col:
                sid = existing_col(strong, ["segment_id", "segment", "id"])
                if sid:
                    sids = normalize_ids(strong[sid])
                    findings["strong_targets_file"]["unique_segment_ids"] = int(
                        sids.nunique(dropna=True)
                    )
        except Exception as exc:
            warnings.append(f"Could not audit strong_targets.csv: {exc}")

    if optional_paths["target_summary"].exists():
        try:
            summary = read_csv(optional_paths["target_summary"])
            findings["target_summary_file"] = {
                "rows": int(len(summary)),
                "columns": list(summary.columns),
            }
        except Exception as exc:
            warnings.append(f"Could not audit target_summary.csv: {exc}")

    if optional_paths["recording_weak_labels_clean"].exists():
        try:
            weak = read_csv(optional_paths["recording_weak_labels_clean"])
            findings["recording_weak_labels_clean"] = {
                "rows": int(len(weak)),
                "columns": list(weak.columns),
            }

            weak_id = existing_col(
                weak, ["recording_id", "audiofileid", "filename", "file"]
            )
            if weak_id:
                ids = normalize_ids(weak[weak_id])
                findings["recording_weak_labels_clean"]["id_column"] = weak_id
                findings["recording_weak_labels_clean"]["missing_ids"] = int(ids.isna().sum())
                findings["recording_weak_labels_clean"]["duplicate_ids"] = int(
                    ids.dropna().duplicated().sum()
                )

                passed, failed = check(
                    int(ids.isna().sum()) == 0,
                    passed, failed,
                    f"Clean weak-label table has no missing {weak_id} values",
                )
                passed, failed = check(
                    int(ids.dropna().duplicated().sum()) == 0,
                    passed, failed,
                    f"Clean weak-label table has unique {weak_id} values",
                )
        except Exception as exc:
            warnings.append(
                f"Could not audit recording_weak_labels_clean.csv: {exc}"
            )

    # ------------------------------------------------------------------
    # Embedding cross-check
    # ------------------------------------------------------------------
    log("")
    log("Cross-checking embeddings...")
    try:
        emb = np.load(embeddings_path, mmap_mode="r")
        findings["embeddings"] = {
            "shape": list(emb.shape),
            "dtype": str(emb.dtype),
        }

        passed, failed = check(
            emb.ndim == 2 and emb.shape[1] == EXPECTED_EMBED_DIM,
            passed, failed,
            f"embeddings.npy has expected shape (*, {EXPECTED_EMBED_DIM})",
        )
        passed, failed = check(
            emb.shape[0] == len(targets),
            passed, failed,
            f"Embedding rows match target rows ({emb.shape[0]} == {len(targets)})",
        )
    except Exception as exc:
        failed += 1
        errors.append(f"Could not load embeddings.npy: {exc}")
        log(f"[FAIL] Could not load embeddings.npy: {exc}")

    try:
        emb_meta = read_csv(emb_meta_path)
        emb_id_col = existing_col(emb_meta, ["segment_id", "segment", "id"])
        passed, failed = check(
            emb_id_col is not None,
            passed, failed,
            f"Embedding metadata contains segment_id (found {emb_id_col})",
        )

        if emb_id_col and target_id_col:
            emb_ids = normalize_ids(emb_meta[emb_id_col])
            exact = target_ids.fillna("<NA>").reset_index(drop=True).equals(
                emb_ids.fillna("<NA>").reset_index(drop=True)
            )
            passed, failed = check(
                exact,
                passed, failed,
                "segment_targets segment_id order exactly matches embedding_metadata.csv",
            )
    except Exception as exc:
        failed += 1
        errors.append(f"Could not read embedding_metadata.csv: {exc}")
        log(f"[FAIL] Could not read embedding_metadata.csv: {exc}")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    status = "PASS" if failed == 0 else "FAIL"

    findings["status"] = status
    findings["summary"] = {
        "passed_checks": passed,
        "failed_checks": failed,
        "warning_count": len(warnings),
        "error_count": len(errors),
    }
    findings["warnings"] = warnings
    findings["errors"] = errors
    findings["audit_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    text_lines = [
        "RANA / SEGMENT TARGET DATASET AUDIT",
        "=" * 78,
        f"Status: {status}",
        f"Audit time UTC: {findings['audit_timestamp_utc']}",
        "",
        f"Target rows:     {len(targets)}",
        f"Metadata rows:   {len(metadata)}",
        f"Target columns:  {len(target_cols)}",
        "",
        f"Passed checks:   {passed}",
        f"Failed checks:   {failed}",
        f"Warnings:        {len(warnings)}",
        "",
    ]

    if "multilabel" in findings:
        m = findings["multilabel"]
        text_lines += [
            "MULTILABEL SUMMARY",
            "-" * 78,
            f"Segments with 0 positives:   {m['segments_with_zero_positive_labels']}",
            f"Segments with 1 positive:    {m['segments_with_one_positive_label']}",
            f"Segments with >1 positives: {m['segments_with_multiple_positive_labels']}",
            f"Maximum positives/segment:  {m['maximum_positive_labels_in_segment']}",
            f"Mean positives/segment:     {m['mean_positive_labels_per_segment']:.6f}",
            f"All targets unknown:         {m['segments_with_all_targets_unknown']}",
            "",
        ]

    if "species_prevalence" in findings:
        text_lines += ["SPECIES PREVALENCE", "-" * 78]
        for row in findings["species_prevalence"]:
            prev = row["prevalence_among_known"]
            prev_s = f"{prev:.6f}" if math.isfinite(prev) else "NA"
            text_lines.append(
                f"{row['species']}: +{row['positive_segments']}, "
                f"0={row['negative_segments']}, "
                f"unknown={row['unknown_segments']}, "
                f"prevalence={prev_s}"
            )
        text_lines.append("")

    if warnings:
        text_lines += ["WARNINGS", "-" * 78]
        text_lines += [f"- {x}" for x in warnings]
        text_lines.append("")

    if errors:
        text_lines += ["ERRORS", "-" * 78]
        text_lines += [f"- {x}" for x in errors]
        text_lines.append("")

    txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(findings, indent=2, ensure_ascii=False, default=safe_json),
        encoding="utf-8",
    )

    log("")
    log("=" * 78)
    log(f"AUDIT STATUS: {status}")
    log(f"Passed checks: {passed}")
    log(f"Failed checks: {failed}")
    log(f"Warnings:      {len(warnings)}")
    log(f"JSON report:   {json_path}")
    log(f"TXT report:    {txt_path}")
    log("=" * 78)

    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(
        description="Audit RANA segment_targets.csv and its alignment with metadata/embeddings."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_ROOT),
        help="Dataset root.",
    )
    parser.add_argument("--targets", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    return audit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
