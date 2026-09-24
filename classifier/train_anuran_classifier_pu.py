#!/usr/bin/env python
"""
Research-grade PU multilabel anuran classifier.

Purpose
-------
Train a positive-unlabeled (PU) multilabel classifier on BirdNET embeddings
without treating every unlabeled target cell as a confirmed biological
negative.

Design
------
* 1024-D BirdNET embeddings
* 42 multilabel species targets
* four-site leave-one-site-out (LOSO) spatial evaluation
* recording-grouped validation for threshold selection
* strong_event positives are the labeled-positive set
* unlabeled strong_event segments are the unlabeled set
* weak_candidate segments are excluded from supervised fitting
* PU bagging:
    For each species, repeatedly sample unlabeled examples as provisional
    negatives, fit a balanced logistic-regression classifier, and average
    probabilities across bags.
  This is a practical PU baseline rather than a formal unbiased PU-risk
  estimator. It avoids declaring every unlabeled example negative while
  still allowing a standard discriminative model to be trained.
* validation thresholds are selected only from the training sites
* held-out site is never used for model/threshold fitting
* species are stratified into:
    standard_support: >= 50 training positives
    low_support:       1-49 training positives
    spatially_novel:    0 training positives
    no_test_positives: 0 test positives
* standard/low support metrics are reported separately from spatial novelty
* output format is compatible with comparison to the existing baseline.

Important interpretation
------------------------
This script does NOT turn NaN into confirmed absence.

For each species:
    P = strong_event segments with target == 1
    U = strong_event segments with target != 1 / unlabeled
The U set is sampled as provisional negatives independently in each PU bag.
Therefore the model is less dependent on the assumption that all unlabeled
segments are true negatives, but this remains a pragmatic PU approximation.

Requirements
------------
numpy, pandas, scikit-learn

Default paths
-------------
embeddings:
    D:\\Acoustics\\AnuraSet_3sec_all\\birdnet_embeddings\\embeddings.npy
embedding metadata:
    D:\\Acoustics\\AnuraSet_3sec_all\\birdnet_embeddings\\embedding_metadata.csv
metadata:
    D:\\Acoustics\\AnuraSet_3sec_all\\metadata.csv
targets:
    D:\\Acoustics\\AnuraSet_3sec_all\\targets\\strong_targets.csv
output:
    D:\\Acoustics\\AnuraSet_3sec_all\\classifier_results_pu

Example
-------
python train_anuran_classifier_pu.py
python train_anuran_classifier_pu.py --targets "D:\\path\\strong_targets(3).csv"
python train_anuran_classifier_pu.py --n-bags 10 --unlabeled-ratio 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

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


DEFAULT_EMBEDDINGS = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\birdnet_embeddings\embeddings.npy"
)
DEFAULT_EMBED_METADATA = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\birdnet_embeddings\embedding_metadata.csv"
)
DEFAULT_METADATA = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\metadata.csv"
)
DEFAULT_TARGETS = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\targets\strong_targets.csv"
)
DEFAULT_OUTPUT = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\classifier_results_pu"
)

MIN_STANDARD_SUPPORT = 50
RANDOM_SEED = 42


def parse_args():
    p = argparse.ArgumentParser(
        description="PU-aware multilabel anuran classifier with four-site LOSO."
    )
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--embedding-metadata", type=Path, default=DEFAULT_EMBED_METADATA)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument(
        "--n-bags",
        type=int,
        default=10,
        help="Number of PU bags per species (default: 10).",
    )
    p.add_argument(
        "--unlabeled-ratio",
        type=float,
        default=2.0,
        help="Provisional negatives sampled per positive in each bag (default: 2.0).",
    )
    p.add_argument(
        "--C",
        type=float,
        default=0.1,
        help="Logistic-regression regularization strength (default: 0.1).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="Maximum logistic-regression iterations.",
    )
    p.add_argument(
        "--min-positive-train",
        type=int,
        default=1,
        help="Minimum positives required to fit a species model.",
    )
    return p.parse_args()


def fail(msg: str):
    raise RuntimeError(msg)


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(f"{name} is missing required columns: {missing}")


def load_and_validate(args):
    print("=" * 72)
    print("RESEARCH-GRADE PU MULTILABEL ANURAN CLASSIFIER")
    print("=" * 72)

    X = np.load(args.embeddings, mmap_mode="r")
    emb_meta = pd.read_csv(args.embedding_metadata)
    metadata = pd.read_csv(args.metadata)
    targets = pd.read_csv(args.targets)

    require_columns(emb_meta, ["segment_id"], "embedding metadata")
    require_columns(metadata, ["segment_id", "recording_id", "site_id", "split", "source_type"], "metadata")
    require_columns(targets, ["segment_id", "recording_id", "site_id", "split", "source_type"], "targets")

    if X.ndim != 2 or X.shape[1] != 1024:
        fail(f"Expected embeddings shape (n, 1024), got {X.shape}")

    if len(X) != len(emb_meta):
        fail(f"Embedding rows ({len(X)}) != embedding metadata rows ({len(emb_meta)})")

    if emb_meta["segment_id"].duplicated().any():
        fail("Duplicate segment_id in embedding metadata.")

    if metadata["segment_id"].duplicated().any():
        fail("Duplicate segment_id in metadata.")

    if targets["segment_id"].duplicated().any():
        fail("Duplicate segment_id in targets.")

    # Exact alignment checks.
    if not np.array_equal(
        emb_meta["segment_id"].astype(str).to_numpy(),
        metadata["segment_id"].astype(str).to_numpy(),
    ):
        fail("Embedding metadata segment_id order does not exactly match metadata.")

    if not np.array_equal(
        targets["segment_id"].astype(str).to_numpy(),
        metadata["segment_id"].astype(str).to_numpy(),
    ):
        fail("Target segment_id order does not exactly match metadata.")

    species_cols = [
        c for c in targets.columns
        if c not in {"segment_id", "recording_id", "site_id", "split", "source_type"}
    ]

    if len(species_cols) != 42:
        print(f"WARNING: detected {len(species_cols)} target columns, expected 42.")

    for c in species_cols:
        vals = targets[c]
        invalid = vals.notna() & ~vals.isin([0, 1, 1.0])
        if invalid.any():
            fail(f"Invalid values in target column {c}: {vals[invalid].unique()[:10]}")

    # Verify target identifiers agree with metadata.
    for c in ["recording_id", "site_id", "split", "source_type"]:
        a = metadata[c].astype(str).to_numpy()
        b = targets[c].astype(str).to_numpy()
        if not np.array_equal(a, b):
            fail(f"Target column {c!r} does not exactly match metadata.")

    print(f"Embeddings: {X.shape}")
    print(f"Segments:   {len(metadata):,}")
    print(f"Species:    {len(species_cols)}")
    print(f"Sites:      {sorted(metadata['site_id'].dropna().unique().tolist())}")
    print()
    print("PU formulation:")
    print("  P = strong_event segments with target == 1")
    print("  U = unlabeled strong_event segments")
    print("  weak_candidate segments excluded from supervised fitting")
    print("  U is sampled as provisional negatives independently per PU bag")
    print()

    return X, emb_meta, metadata, targets, species_cols


def grouped_validation_split(df, seed=42):
    """
    Deterministic recording-level split.

    The split is stratified approximately by recording, not by individual
    segment, preventing overlapping segments from leaking across train/val.
    """
    rng = np.random.default_rng(seed)
    recordings = df["recording_id"].astype(str).drop_duplicates().to_numpy()
    rng.shuffle(recordings)

    n_val = max(1, int(round(len(recordings) * 0.20)))
    val_recordings = set(recordings[:n_val])

    val_mask = df["recording_id"].astype(str).isin(val_recordings).to_numpy()
    if val_mask.sum() == 0 or (~val_mask).sum() == 0:
        fail("Grouped validation split produced an empty partition.")
    return ~val_mask, val_mask


def choose_threshold(y_true, scores):
    """
    Select threshold maximizing F1 on validation data.

    If validation has no positives, return 0.5. The threshold is never
    selected using the held-out site.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if y_true.sum() == 0:
        return 0.5

    # Candidate thresholds from observed scores plus endpoints.
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return 0.5

    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
                finite,
            ]
        )
    )

    best_t = 0.5
    best_f1 = -1.0

    for t in candidates:
        pred = (scores >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t


def safe_metric(fn, *args):
    try:
        return float(fn(*args))
    except ValueError:
        return np.nan


def fit_pu_bagged_model(
    X_train,
    y_train,
    n_bags,
    unlabeled_ratio,
    C,
    max_iter,
    seed,
):
    """
    Fit a bagged PU logistic-regression ensemble.

    Positives are always retained.
    Each bag samples a different subset of U as provisional negatives.

    Returns:
        list of fitted LogisticRegression models
    """
    y_train = np.asarray(y_train).astype(int)

    pos_idx = np.flatnonzero(y_train == 1)
    unlabeled_idx = np.flatnonzero(y_train == 0)

    if len(pos_idx) == 0:
        return []

    if len(unlabeled_idx) == 0:
        fail("No unlabeled examples available for PU training.")

    n_neg = max(1, int(round(len(pos_idx) * unlabeled_ratio)))
    n_neg = min(n_neg, len(unlabeled_idx))

    rng = np.random.default_rng(seed)
    models = []

    for bag in range(n_bags):
        # Sampling with replacement across bags; without replacement within a bag.
        neg_idx = rng.choice(unlabeled_idx, size=n_neg, replace=False)

        idx = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(idx)

        model = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed + bag,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train[idx], y_train[idx])

        models.append(model)

    return models


def predict_pu(models, X):
    if not models:
        return np.full(len(X), np.nan, dtype=float)

    probs = np.vstack([m.predict_proba(X)[:, 1] for m in models])
    return np.mean(probs, axis=0)


def species_status(train_positive, test_positive):
    if test_positive == 0:
        return "no_test_positives"
    if train_positive == 0:
        return "spatially_novel"
    if train_positive >= MIN_STANDARD_SUPPORT:
        return "standard_support"
    return "low_support"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    X, emb_meta, metadata, targets, species_cols = load_and_validate(args)

    sites = sorted(metadata["site_id"].dropna().astype(str).unique().tolist())
    all_fold_rows = []
    all_species_rows = []
    all_prediction_rows = []

    run_config = {
        "method": "PU_bagged_logistic_regression",
        "random_seed": RANDOM_SEED,
        "n_bags": args.n_bags,
        "unlabeled_ratio": args.unlabeled_ratio,
        "C": args.C,
        "max_iter": args.max_iter,
        "min_standard_support": MIN_STANDARD_SUPPORT,
        "embeddings": str(args.embeddings),
        "embedding_metadata": str(args.embedding_metadata),
        "metadata": str(args.metadata),
        "targets": str(args.targets),
        "output": str(args.output),
        "n_segments": int(len(metadata)),
        "embedding_dimension": int(X.shape[1]),
        "n_species": int(len(species_cols)),
        "sites": sites,
    }

    (args.output / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    for held_out_site in sites:
        print("-" * 72)
        print(f"HELD-OUT SITE: {held_out_site}")

        train_site_mask = metadata["site_id"].astype(str).ne(held_out_site).to_numpy()
        test_site_mask = metadata["site_id"].astype(str).eq(held_out_site).to_numpy()

        # Only strong_event segments participate in PU supervised fitting.
        strong_train_mask = train_site_mask & metadata["source_type"].eq("strong_event").to_numpy()
        strong_test_mask = test_site_mask & metadata["source_type"].eq("strong_event").to_numpy()

        train_idx_all = np.flatnonzero(strong_train_mask)
        test_idx_all = np.flatnonzero(strong_test_mask)

        if len(train_idx_all) == 0:
            fail(f"No strong_event training segments for held-out site {held_out_site}.")

        # Recording-grouped validation split is made ONLY inside training sites.
        train_df = metadata.iloc[train_idx_all].copy()
        fit_rel_mask, val_rel_mask = grouped_validation_split(
            train_df, seed=RANDOM_SEED + sites.index(held_out_site)
        )

        fit_idx = train_idx_all[fit_rel_mask]
        val_idx = train_idx_all[val_rel_mask]

        # Track fold species support before fitting.
        status_rows = []

        for species_i, species in enumerate(species_cols):
            y_all = targets[species].notna() & targets[species].eq(1)

            train_positive = int(y_all.iloc[train_idx_all].sum())
            test_positive = int(y_all.iloc[test_idx_all].sum())
            status = species_status(train_positive, test_positive)

            status_rows.append(
                {
                    "held_out_site": held_out_site,
                    "species": species,
                    "train_positive_segments": train_positive,
                    "test_positive_segments": test_positive,
                    "status": status,
                }
            )

        status_df = pd.DataFrame(status_rows)
        print(status_df["status"].value_counts().to_string())

        fold_predictions = {
            "segment_id": metadata.iloc[test_idx_all]["segment_id"].astype(str).to_numpy(),
            "recording_id": metadata.iloc[test_idx_all]["recording_id"].astype(str).to_numpy(),
            "site_id": np.repeat(held_out_site, len(test_idx_all)),
        }

        evaluated_rows = []

        for species_i, species in enumerate(species_cols):
            y_positive = targets[species].notna() & targets[species].eq(1)
            y = y_positive.astype(int).to_numpy()

            train_positive = int(y[train_idx_all].sum())
            test_positive = int(y[test_idx_all].sum())

            status = species_status(train_positive, test_positive)

            # Spatially novel or no-test-positive species do not contribute
            # conventional supervised metrics.
            if status in {"spatially_novel", "no_test_positives"}:
                continue

            y_fit = y[fit_idx]
            y_val = y[val_idx]
            y_test = y[test_idx_all]

            if y_fit.sum() < args.min_positive_train:
                continue

            # If validation happens to have no positives, threshold defaults to 0.5.
            models = fit_pu_bagged_model(
                X[fit_idx],
                y_fit,
                n_bags=args.n_bags,
                unlabeled_ratio=args.unlabeled_ratio,
                C=args.C,
                max_iter=args.max_iter,
                seed=RANDOM_SEED + species_i * 1000 + sites.index(held_out_site) * 100,
            )

            if not models:
                continue

            val_scores = predict_pu(models, X[val_idx])
            test_scores = predict_pu(models, X[test_idx_all])

            threshold = choose_threshold(y_val, val_scores)
            test_pred = (test_scores >= threshold).astype(int)

            ap = safe_metric(average_precision_score, y_test, test_scores)
            auc = safe_metric(roc_auc_score, y_test, test_scores)

            row = {
                "held_out_site": held_out_site,
                "species": species,
                "status": status,
                "train_positive_segments": train_positive,
                "test_positive_segments": test_positive,
                "validation_positive_segments": int(y_val.sum()),
                "fit_positive_segments": int(y_fit.sum()),
                "pu_bags": int(len(models)),
                "unlabeled_ratio": float(args.unlabeled_ratio),
                "threshold": float(threshold),
                "precision": precision_score(y_test, test_pred, zero_division=0),
                "recall": recall_score(y_test, test_pred, zero_division=0),
                "f1": f1_score(y_test, test_pred, zero_division=0),
                "average_precision": ap,
                "roc_auc": auc,
            }
            evaluated_rows.append(row)

            fold_predictions[f"{species}__score"] = test_scores
            fold_predictions[f"{species}__pred"] = test_pred

        fold_species_df = pd.DataFrame(evaluated_rows)
        all_species_rows.extend(evaluated_rows)

        if not fold_species_df.empty:
            summary_row = {
                "held_out_site": held_out_site,
                "n_species": int(len(fold_species_df)),
                "macro_f1": float(fold_species_df["f1"].mean()),
                "macro_precision": float(fold_species_df["precision"].mean()),
                "macro_recall": float(fold_species_df["recall"].mean()),
                "macro_average_precision": float(fold_species_df["average_precision"].mean()),
                "macro_roc_auc": float(fold_species_df["roc_auc"].mean()),
                "median_train_positive_segments": float(
                    fold_species_df["train_positive_segments"].median()
                ),
            }
            all_fold_rows.append(summary_row)

        pred_df = pd.DataFrame(fold_predictions)
        pred_path = args.output / f"predictions_{held_out_site}.csv"
        pred_df.to_csv(pred_path, index=False)

        status_path = args.output / f"species_status_{held_out_site}.csv"
        status_df.to_csv(status_path, index=False)

        print()
        if not fold_species_df.empty:
            print(fold_species_df.to_string(index=False))
        else:
            print("No conventionally evaluable species in this fold.")

    # Fold summary
    summary_df = pd.DataFrame(all_fold_rows)

    if all_species_rows:
        species_df = pd.DataFrame(all_species_rows)
    else:
        species_df = pd.DataFrame()

    if not summary_df.empty:
        metric_cols = [
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "macro_average_precision",
            "macro_roc_auc",
        ]
        all_fold_summary = {
            "held_out_site": "ALL_FOLDS",
            "n_species": int(len(species_df)),
            "macro_f1": float(species_df["f1"].mean()),
            "macro_precision": float(species_df["precision"].mean()),
            "macro_recall": float(species_df["recall"].mean()),
            "macro_average_precision": float(species_df["average_precision"].mean()),
            "macro_roc_auc": float(species_df["roc_auc"].mean()),
            "median_train_positive_segments": float(
                species_df["train_positive_segments"].median()
            ),
        }
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([all_fold_summary])],
            ignore_index=True,
        )

    # Spatial novelty summary.
    status_all = []
    for held_out_site in sites:
        path = args.output / f"species_status_{held_out_site}.csv"
        if path.exists():
            status_all.append(pd.read_csv(path))

    if status_all:
        status_all_df = pd.concat(status_all, ignore_index=True)
        novel_df = status_all_df[status_all_df["status"] == "spatially_novel"].copy()

        novelty_summary = (
            novel_df.groupby("held_out_site", as_index=False)
            .agg(
                spatially_novel_species=("species", "nunique"),
                novel_test_positive_segments=("test_positive_segments", "sum"),
            )
        )
        novelty_summary.to_csv(
            args.output / "spatial_novelty_summary.csv",
            index=False,
        )

    summary_df.to_csv(args.output / "loso_summary.csv", index=False)
    species_df.to_csv(args.output / "loso_species_results.csv", index=False)

    # Explicit support-stratified tables.
    if not species_df.empty:
        for status in ["standard_support", "low_support"]:
            species_df[species_df["status"] == status].to_csv(
                args.output / f"{status}_results.csv",
                index=False,
            )

    print()
    print("=" * 72)
    print("PU LOSO SUMMARY")
    print("=" * 72)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No evaluated folds.")

    if status_all:
        print()
        print("Spatially novel species by held-out site:")
        print(novelty_summary.to_string(index=False))

    print()
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("ERROR:", exc)
        raise
