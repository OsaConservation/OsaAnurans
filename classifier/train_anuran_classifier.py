#!/usr/bin/env python3
"""
Research-grade multilabel anuran classifier using BirdNET embeddings.

Design:
- 1024-D BirdNET acoustic v2.4 embeddings as fixed features.
- Strong segment-level labels are the supervised targets.
- Weak recording-level labels are NOT copied to individual segments.
- Four-site leave-one-site-out (LOSO) spatial evaluation.
- Per-fold species eligibility is explicit:
    * shared species: >=1 positive training segment
    * low-support: 1-49 positive training segments
    * standard-support: >=50 positive training segments
    * spatially novel: 0 positive training segments
- Reports are separated for shared/learnable species and spatially novel species.
- Multilabel classifier: one-vs-rest logistic regression with class weighting.
- Thresholds are selected on an inner grouped validation split from TRAINING SITES ONLY.
- Test site is never used for threshold selection/model selection.
- Grouping is by recording_id to prevent adjacent 3-s segments from crossing train/validation.
- Optional probability calibration is intentionally omitted from the first baseline;
  threshold selection is performed per species on validation predictions.

Expected default files:
    D:\\Acoustics\\AnuraSet_3sec_all\\birdnet_embeddings\\embeddings.npy
    D:\\Acoustics\\AnuraSet_3sec_all\\birdnet_embeddings\\embedding_metadata.csv
    D:\\Acoustics\\AnuraSet_3sec_all\\metadata.csv
    D:\\Acoustics\\AnuraSet_3sec_all\\targets\\strong_targets.csv

Usage:
    python train_anuran_classifier.py

Optional:
    python train_anuran_classifier.py --data-root D:\\Acoustics\\AnuraSet_3sec_all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path(r"D:\Acoustics\AnuraSet_3sec_all")
DEFAULT_EMBEDDINGS = DEFAULT_ROOT / "birdnet_embeddings" / "embeddings.npy"
DEFAULT_EMBED_METADATA = DEFAULT_ROOT / "birdnet_embeddings" / "embedding_metadata.csv"
DEFAULT_METADATA = DEFAULT_ROOT / "metadata.csv"
DEFAULT_TARGETS = DEFAULT_ROOT / "targets" / "strong_targets.csv"
DEFAULT_OUT = DEFAULT_ROOT / "classifier_results"


META_COLUMNS = {
    "segment_id",
    "recording_id",
    "site_id",
    "split",
    "source_type",
    "recording_date",
    "recording_time",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--embeddings", type=Path, default=None)
    p.add_argument("--embedding-metadata", type=Path, default=None)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--targets", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--validation-size", type=float, default=0.20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--min-train-positives",
        type=int,
        default=1,
        help="Minimum training positives for a species to be included in the "
             "learnable/shared-species evaluation. Default: 1.",
    )
    p.add_argument(
        "--standard-support",
        type=int,
        default=50,
        help="Training-positive threshold used to distinguish low-support "
             "from standard-support species. Default: 50.",
    )
    return p.parse_args()


def resolve_paths(args: argparse.Namespace):
    root = args.data_root
    embeddings = args.embeddings or (root / "birdnet_embeddings" / "embeddings.npy")
    embedding_metadata = (
        args.embedding_metadata
        or (root / "birdnet_embeddings" / "embedding_metadata.csv")
    )
    metadata = args.metadata or (root / "metadata.csv")
    targets = args.targets or (root / "targets" / "strong_targets.csv")
    output = args.output or (root / "classifier_results")
    return embeddings, embedding_metadata, metadata, targets, output


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def read_inputs(
    embeddings_path: Path,
    embedding_metadata_path: Path,
    metadata_path: Path,
    targets_path: Path,
):
    X = np.load(embeddings_path)
    emb_meta = pd.read_csv(embedding_metadata_path)
    metadata = pd.read_csv(metadata_path)
    targets = pd.read_csv(targets_path)

    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D; got {X.shape}")
    if X.shape[1] != 1024:
        raise ValueError(f"Expected 1024-D BirdNET embeddings; got {X.shape[1]}")
    if len(X) != len(emb_meta):
        raise ValueError("Embedding rows do not match embedding metadata rows.")
    if "segment_id" not in emb_meta:
        raise ValueError("embedding_metadata.csv lacks segment_id.")
    if "segment_id" not in metadata:
        raise ValueError("metadata.csv lacks segment_id.")
    if "segment_id" not in targets:
        raise ValueError("targets file lacks segment_id.")

    if emb_meta["segment_id"].duplicated().any():
        raise ValueError("Duplicate segment_id in embedding metadata.")
    if metadata["segment_id"].duplicated().any():
        raise ValueError("Duplicate segment_id in metadata.")
    if targets["segment_id"].duplicated().any():
        raise ValueError("Duplicate segment_id in targets.")

    # Exact identity/order check.
    if not np.array_equal(
        emb_meta["segment_id"].astype(str).to_numpy(),
        metadata["segment_id"].astype(str).to_numpy(),
    ):
        raise ValueError(
            "Embedding metadata segment_id order does not exactly match metadata.csv."
        )

    if not np.array_equal(
        emb_meta["segment_id"].astype(str).to_numpy(),
        targets["segment_id"].astype(str).to_numpy(),
    ):
        raise ValueError(
            "Embedding metadata segment_id order does not exactly match targets."
        )

    # Required grouping/site fields.
    for c in ["recording_id", "site_id", "source_type"]:
        if c not in metadata:
            raise ValueError(f"metadata.csv lacks required column: {c}")

    # Strong target matrix: exclude all metadata/summary columns.
    target_cols = [
        c for c in targets.columns
        if c not in META_COLUMNS
        and not c.startswith("SPECIES_")
        and c not in {"n_strong_species"}
    ]

    if not target_cols:
        raise ValueError("No species target columns found.")

    # Only binary positive or unknown are permitted.
    bad = {}
    for c in target_cols:
        vals = targets[c].dropna().unique()
        invalid = [v for v in vals if v != 1]
        if invalid:
            bad[c] = invalid
    if bad:
        raise ValueError(f"Invalid target values: {bad}")

    return X.astype(np.float32, copy=False), emb_meta, metadata, targets, target_cols


def make_training_matrix(
    X: np.ndarray,
    metadata: pd.DataFrame,
    targets: pd.DataFrame,
    species: str,
    train_idx: np.ndarray,
    fit_idx: np.ndarray,
):
    """
    Strong labels are positive-only annotations.

    For each species:
      y=1 for confirmed strong positives.
      y=0 for segments with an explicit strong-negative annotation is NOT
      available in this dataset.

    Therefore, this baseline trains on positives plus unlabeled examples
    treated as negatives ONLY within the training population. This is a
    pragmatic one-vs-rest baseline, not a claim that NaN means biological
    absence.

    To make the distinction explicit, weak_candidate rows are excluded from
    the supervised training baseline. Only strong_event rows participate.
    """
    strong_mask = metadata["source_type"].eq("strong_event").to_numpy()
    train_strong = train_idx[strong_mask[train_idx]]
    fit_strong = fit_idx[strong_mask[fit_idx]]

    y_all = targets[species].eq(1).to_numpy()

    # Training classifier uses all strong-event segments:
    # positives = confirmed species presence
    # unlabeled strong-event segments = negative proxy.
    # This is documented and should be replaced by PU learning if desired.
    y_train = y_all[train_strong].astype(int)

    return train_strong, fit_strong, y_train


def safe_ap(y_true, score):
    if np.sum(y_true) == 0:
        return np.nan
    if np.sum(y_true) == len(y_true):
        return np.nan
    return float(average_precision_score(y_true, score))


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def select_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """
    Select threshold maximizing F1 on validation data.

    Threshold selection is restricted to the training-site validation split.
    """
    if y_true.sum() == 0:
        return 0.5

    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 19),
                np.quantile(prob, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    candidates = np.clip(candidates, 0.01, 0.99)

    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        pred = prob >= t
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def fit_binary_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    random_state: int,
    C: float,
    max_iter: int,
):
    if np.sum(y_train == 1) == 0:
        return None, None, None

    if np.sum(y_train == 0) == 0:
        return None, None, None

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)

    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(Xtr, y_train)

    p = clf.predict_proba(Xv)[:, 1]
    return scaler, clf, p


def evaluate_predictions(y_true, prob, threshold):
    pred = prob >= threshold
    return {
        "n_test": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "average_precision": safe_ap(y_true, prob),
        "roc_auc": safe_auc(y_true, prob),
        "predicted_positive": int(pred.sum()),
    }


def run_fold(
    X,
    metadata,
    targets,
    species,
    heldout_site,
    args,
):
    site = metadata["site_id"].astype(str).to_numpy()
    groups = metadata["recording_id"].astype(str).to_numpy()
    source = metadata["source_type"].astype(str).to_numpy()
    y = targets[species].eq(1).to_numpy()

    test_idx = np.flatnonzero(site == heldout_site)
    train_idx = np.flatnonzero(site != heldout_site)

    # Strong-only training population.
    train_strong = train_idx[source[train_idx] == "strong_event"]
    test_strong = test_idx[source[test_idx] == "strong_event"]

    train_pos = int(y[train_strong].sum())
    test_pos = int(y[test_strong].sum())

    if test_pos == 0:
        return {
            "held_out_site": heldout_site,
            "species": species,
            "status": "no_test_positives",
            "train_positive_segments": train_pos,
            "test_positive_segments": test_pos,
        }, None

    if train_pos < args.min_train_positives:
        # Spatially novel if zero; otherwise not enough support for configured baseline.
        status = "spatially_novel" if train_pos == 0 else "insufficient_training_support"
        return {
            "held_out_site": heldout_site,
            "species": species,
            "status": status,
            "train_positive_segments": train_pos,
            "test_positive_segments": test_pos,
        }, None

    # Grouped validation split inside training sites.
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_size,
        random_state=args.random_state,
    )
    train_sub_pos, val_sub_pos = next(
        gss.split(train_strong, y[train_strong], groups[train_strong])
    )
    fit_idx = train_strong[train_sub_pos]
    val_idx = train_strong[val_sub_pos]

    y_fit = y[fit_idx].astype(int)
    y_val = y[val_idx].astype(int)
    y_test = y[test_strong].astype(int)

    if y_fit.sum() == 0 or y_fit.sum() == len(y_fit):
        return {
            "held_out_site": heldout_site,
            "species": species,
            "status": "validation_training_degenerate",
            "train_positive_segments": train_pos,
            "test_positive_segments": test_pos,
        }, None

    scaler, clf, val_prob = fit_binary_model(
        X[fit_idx],
        y_fit,
        X[val_idx],
        args.random_state,
        args.C,
        args.max_iter,
    )
    if clf is None:
        return {
            "held_out_site": heldout_site,
            "species": species,
            "status": "model_fit_failed",
            "train_positive_segments": train_pos,
            "test_positive_segments": test_pos,
        }, None

    threshold = select_threshold(y_val, val_prob)

    Xtest = scaler.transform(X[test_strong])
    test_prob = clf.predict_proba(Xtest)[:, 1]
    metrics = evaluate_predictions(y_test, test_prob, threshold)

    result = {
        "held_out_site": heldout_site,
        "species": species,
        "status": (
            "standard_support"
            if train_pos >= args.standard_support
            else "low_support"
        ),
        "train_positive_segments": train_pos,
        "test_positive_segments": test_pos,
        "training_strong_segments": int(len(train_strong)),
        "validation_segments": int(len(val_idx)),
        **metrics,
    }

    # Store per-segment predictions for auditability.
    pred_df = pd.DataFrame({
        "segment_id": metadata.iloc[test_strong]["segment_id"].astype(str).values,
        "recording_id": metadata.iloc[test_strong]["recording_id"].astype(str).values,
        "site_id": heldout_site,
        "species": species,
        "y_true": y_test,
        "probability": test_prob,
        "prediction": (test_prob >= threshold).astype(int),
        "threshold": threshold,
    })
    return result, pred_df


def aggregate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    usable = results[
        results["status"].isin(["standard_support", "low_support"])
    ].copy()

    rows = []
    for site, g in usable.groupby("held_out_site"):
        rows.append({
            "held_out_site": site,
            "n_species": len(g),
            "macro_f1": g["f1"].mean(),
            "macro_precision": g["precision"].mean(),
            "macro_recall": g["recall"].mean(),
            "macro_average_precision": g["average_precision"].mean(),
            "macro_roc_auc": g["roc_auc"].mean(),
            "median_train_positive_segments": g["train_positive_segments"].median(),
        })

    if len(usable):
        rows.append({
            "held_out_site": "ALL_FOLDS",
            "n_species": len(usable),
            "macro_f1": usable["f1"].mean(),
            "macro_precision": usable["precision"].mean(),
            "macro_recall": usable["recall"].mean(),
            "macro_average_precision": usable["average_precision"].mean(),
            "macro_roc_auc": usable["roc_auc"].mean(),
            "median_train_positive_segments": usable["train_positive_segments"].median(),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    emb_path, emb_meta_path, metadata_path, targets_path, outdir = resolve_paths(args)

    for p, d in [
        (emb_path, "Embeddings"),
        (emb_meta_path, "Embedding metadata"),
        (metadata_path, "Metadata"),
        (targets_path, "Strong targets"),
    ]:
        require_file(p, d)

    outdir.mkdir(parents=True, exist_ok=True)

    X, emb_meta, metadata, targets, species = read_inputs(
        emb_path, emb_meta_path, metadata_path, targets_path
    )

    sites = sorted(metadata["site_id"].astype(str).unique())

    print("=" * 72)
    print("RESEARCH-GRADE MULTILABEL ANURAN CLASSIFIER")
    print("=" * 72)
    print(f"Embeddings: {X.shape}")
    print(f"Segments:   {len(metadata):,}")
    print(f"Species:    {len(species)}")
    print(f"Sites:      {sites}")
    print()
    print("Important: strong labels are positive-only annotations.")
    print("Unlabeled strong-event segments are used as negative proxies in")
    print("this baseline; weak_candidate segments are excluded from fitting.")
    print("Do NOT interpret NaN as confirmed biological absence.")
    print()

    all_results = []
    all_predictions = []

    for heldout in sites:
        print("-" * 72)
        print(f"HELD-OUT SITE: {heldout}")

        fold_results = []
        for sp in species:
            result, preds = run_fold(X, metadata, targets, sp, heldout, args)
            all_results.append(result)
            fold_results.append(result)
            if preds is not None:
                all_predictions.append(preds)

        fold_df = pd.DataFrame(fold_results)
        print(
            fold_df["status"].value_counts(dropna=False).to_string()
        )

    results_df = pd.DataFrame(all_results)
    predictions_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )

    aggregates = aggregate_metrics(results_df)

    # Spatial novelty summary.
    novel = results_df[results_df["status"] == "spatially_novel"].copy()
    low = results_df[results_df["status"] == "low_support"].copy()
    standard = results_df[results_df["status"] == "standard_support"].copy()

    novelty_summary = (
        novel.groupby("held_out_site")
        .agg(
            spatially_novel_species=("species", "count"),
            novel_test_positive_segments=("test_positive_segments", "sum"),
        )
        .reset_index()
    )

    # Save outputs.
    results_df.to_csv(outdir / "per_species_loso_results.csv", index=False)
    aggregates.to_csv(outdir / "loso_summary.csv", index=False)
    novelty_summary.to_csv(outdir / "spatial_novelty_summary.csv", index=False)
    novel.to_csv(outdir / "spatially_novel_species.csv", index=False)
    low.to_csv(outdir / "low_support_species.csv", index=False)
    standard.to_csv(outdir / "standard_support_species.csv", index=False)

    if not predictions_df.empty:
        predictions_df.to_csv(
            outdir / "test_segment_predictions.csv", index=False
        )

    config = {
        "embeddings": str(emb_path),
        "embedding_metadata": str(emb_meta_path),
        "metadata": str(metadata_path),
        "targets": str(targets_path),
        "output": str(outdir),
        "C": args.C,
        "max_iter": args.max_iter,
        "validation_size": args.validation_size,
        "random_state": args.random_state,
        "min_train_positives": args.min_train_positives,
        "standard_support": args.standard_support,
        "sites": sites,
        "species": species,
        "n_segments": len(metadata),
        "embedding_dimension": int(X.shape[1]),
    }
    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("LOSO SUMMARY")
    print("=" * 72)
    if len(aggregates):
        print(aggregates.to_string(index=False))
    print()
    print("Spatially novel species by held-out site:")
    print(novelty_summary.to_string(index=False))
    print()
    print(f"Results written to: {outdir}")


if __name__ == "__main__":
    main()
