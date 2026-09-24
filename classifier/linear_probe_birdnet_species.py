#!/usr/bin/env python3
"""
Research-grade species-level linear-probe analysis of BirdNET 2.4 embeddings.

Design:
  - Uses ONLY strong_event segments with a non-empty event_label.
  - Collapses final _L/_M/_H suffixes into species_label; event_label is preserved.
  - Records recording_quality (L/M/H) for later confounder analysis.
  - Uses recording-level grouping to prevent segment leakage.
  - Uses the predefined train/val/test split for the final held-out test.
  - Selects hyperparameters using recording-grouped CV inside TRAIN only.
  - Fits the selected model on train + validation and evaluates once on TEST.
  - Compares multinomial softmax (logistic) regression with a k-nearest-neighbor baseline.
  - Produces publication-ready CSV tables and figures.

Expected files:
  embeddings.npy                         N x 1024 float32
  metadata_with_embedding_status.csv    same N rows as embeddings.npy

Example Windows:
  python linear_probe_birdnet.py ^
    --embedding-file "D:\\Acoustics\\AnuraSet_3sec\\INCT4\\birdnet_embeddings\\embeddings.npy" ^
    --metadata "D:\\Acoustics\\AnuraSet_3sec\\INCT4\\birdnet_embeddings\\metadata_with_embedding_status.csv" ^
    --output-dir "D:\\Acoustics\\AnuraSet_3sec\\INCT4\\birdnet_embeddings\\linear_probe"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-file", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--logistic-c",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
    )
    p.add_argument(
        "--knn-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 11, 15, 21],
    )
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


QUALITY_CODES = {"L", "M", "H"}


def collapse_quality_label(label):
    """Collapse final _L/_M/_H recording-quality suffix into species."""
    if pd.isna(label):
        return np.nan
    label = str(label).strip()
    parts = label.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in QUALITY_CODES:
        return parts[0]
    return label


def add_species_labels(meta):
    """Preserve event_label while adding species_label and recording_quality."""
    meta = meta.copy()
    meta["species_label"] = meta["event_label"].apply(collapse_quality_label)

    def get_quality(label):
        if pd.isna(label):
            return np.nan
        parts = str(label).strip().rsplit("_", 1)
        return parts[1] if len(parts) == 2 and parts[1] in QUALITY_CODES else np.nan

    meta["recording_quality"] = meta["event_label"].apply(get_quality)
    return meta


def load_data(embedding_file: Path, metadata_file: Path):
    X = np.load(embedding_file)
    meta = pd.read_csv(metadata_file)

    if X.ndim != 2 or X.shape[1] != 1024:
        raise ValueError(f"Expected embeddings shape (N, 1024), got {X.shape}")
    if len(X) != len(meta):
        raise ValueError(
            f"Embedding rows ({len(X)}) != metadata rows ({len(meta)})"
        )

    required = {"segment_id", "recording_id", "split", "source_type", "event_label"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if "embedding_status" in meta.columns:
        ok = meta["embedding_status"].astype(str).eq("success").to_numpy()
        X = X[ok]
        meta = meta.loc[ok].reset_index(drop=True)

    # The classifier target is deliberately event_label, not
    # weak_recording_labels, because the latter are recording-level labels.
    strong = meta["source_type"].astype(str).eq("strong_event")
    labels = meta["event_label"].fillna("").astype(str).str.strip()
    labeled = labels.ne("")

    keep = strong & labeled
    X = X[keep.to_numpy()]
    meta = meta.loc[keep].reset_index(drop=True)
    meta = add_species_labels(meta)
    meta["target"] = meta["species_label"].astype(str)

    # Verify the predefined split is recording-level.
    split_counts = meta.groupby("recording_id")["split"].nunique()
    bad = split_counts[split_counts > 1]
    if len(bad):
        raise ValueError(
            "Recording leakage detected: at least one recording occurs in "
            f"multiple splits. Examples: {bad.head().to_dict()}"
        )

    return X.astype(np.float32), meta


def make_cv(n_splits: int, random_state: int):
    # StratifiedGroupKFold is preferable because it tries to preserve class
    # balance while keeping all segments from a recording together.
    try:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    except Exception:
        return GroupKFold(n_splits=n_splits)



class SoftmaxRegression:
    """Multinomial logistic regression implemented with NumPy/SciPy.

    This intentionally avoids sklearn.linear_model because importing that
    module imports sklearn.svm._libsvm on Windows.
    """

    def __init__(self, C=1.0, max_iter=1000, class_weight="balanced",
                 random_state=42):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)

        # Standardization is done inside the model so every CV fold is
        # independent and no validation information enters preprocessing.
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        Z = (X - self.mean_) / self.scale_

        # Balanced class weights: n / (K * class_count).
        sample_w = np.ones(n, dtype=np.float64)
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=k).astype(np.float64)
            class_w = n / (k * np.maximum(counts, 1.0))
            sample_w = class_w[y_idx]

        # Bias + weights. Regularize weights but not the intercept.
        Z1 = np.column_stack([np.ones(n), Z])

        # Stable softmax.
        def unpack(theta):
            return theta.reshape(k, d + 1)

        def loss_grad(theta):
            W = unpack(theta)
            scores = Z1 @ W.T
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            logp = -np.log(np.maximum(probs[np.arange(n), y_idx], 1e-15))
            data_loss = np.sum(sample_w * logp) / np.sum(sample_w)

            # L2 penalty on non-intercept terms.
            reg = 0.5 * np.sum(W[:, 1:] ** 2) / self.C
            loss = data_loss + reg / n

            diff = probs
            diff[np.arange(n), y_idx] -= 1.0
            diff *= sample_w[:, None]
            grad = (diff.T @ Z1) / np.sum(sample_w)
            grad[:, 1:] += W[:, 1:] / (self.C * n)

            return float(loss), grad.ravel()

        theta0 = np.zeros(k * (d + 1), dtype=np.float64)
        result = minimize(
            fun=lambda th: loss_grad(th),
            x0=theta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-8, "gtol": 1e-6},
        )
        if not result.success:
            print(f"  warning: softmax optimizer: {result.message}")

        self.coef_ = unpack(result.x)[:, 1:]
        self.intercept_ = unpack(result.x)[:, 0]
        self.n_iter_ = result.nit
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.scale_
        scores = Z @ self.coef_.T + self.intercept_
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def cv_score_logistic(X, y, groups, c_values, n_splits, random_state):
    cv = make_cv(n_splits, random_state)
    rows = []

    for c in c_values:
        fold_scores = []
        for tr, va in cv.split(X, y, groups):
            model = SoftmaxRegression(
                C=c,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            fold_scores.append(
                f1_score(y[va], pred, average="macro", zero_division=0)
            )

        rows.append({
            "model": "multinomial_softmax_regression",
            "hyperparameter": f"C={c:g}",
            "value": c,
            "mean_macro_f1": np.mean(fold_scores),
            "std_macro_f1": (
                np.std(fold_scores, ddof=1) if len(fold_scores) > 1 else 0.0
            ),
            "fold_scores": json.dumps([float(v) for v in fold_scores]),
        })

    return pd.DataFrame(rows)


def cv_score_knn(X, y, groups, k_values, n_splits, random_state):
    cv = make_cv(n_splits, random_state)
    rows = []

    for k in k_values:
        fold_scores = []
        for tr, va in cv.split(X, y, groups):
            model = Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=k,
                    weights="distance",
                    metric="euclidean",
                )),
            ])
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            fold_scores.append(f1_score(y[va], pred, average="macro", zero_division=0))

        rows.append({
            "model": "knn",
            "hyperparameter": f"k={k}",
            "value": k,
            "mean_macro_f1": np.mean(fold_scores),
            "std_macro_f1": np.std(fold_scores, ddof=1) if len(fold_scores) > 1 else 0.0,
            "fold_scores": json.dumps([float(v) for v in fold_scores]),
        })

    return pd.DataFrame(rows)


def evaluate(model, X, y, split_name, outdir):
    pred = model.predict(X)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "n_samples": len(y),
        "n_classes": len(np.unique(y)),
    }

    report = classification_report(
        y, pred, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        outdir / f"classification_report_{split_name}.csv"
    )

    classes = np.unique(np.concatenate([np.asarray(y), np.asarray(pred)]))
    cm = confusion_matrix(y, pred, labels=classes)

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "true"
    cm_df.to_csv(outdir / f"confusion_matrix_{split_name}.csv")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title(f"Linear probe confusion matrix — {split_name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_matrix_{split_name}.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / f"confusion_matrix_{split_name}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    return metrics



def generate_diagnostic_report(meta, outdir):
    """Generate dataset/split diagnostics without changing model fitting."""
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Species x split: segment counts and unique recording counts.
    seg = (
        meta.groupby(["species_label", "split"])
        .size()
        .rename("n_segments")
        .reset_index()
    )
    rec = (
        meta.groupby(["species_label", "split"])["recording_id"]
        .nunique()
        .rename("n_recordings")
        .reset_index()
    )
    split_diag = seg.merge(rec, on=["species_label", "split"], how="outer")
    split_diag = split_diag.sort_values(["species_label", "split"])
    split_diag.to_csv(outdir / "diagnostic_species_split_counts.csv", index=False)

    # Wide versions make the train/val/test imbalance easy to inspect.
    seg_wide = seg.pivot(index="species_label", columns="split", values="n_segments").fillna(0).astype(int)
    rec_wide = rec.pivot(index="species_label", columns="split", values="n_recordings").fillna(0).astype(int)
    for c in ["train", "val", "test"]:
        if c not in seg_wide.columns:
            seg_wide[c] = 0
        if c not in rec_wide.columns:
            rec_wide[c] = 0
    seg_wide = seg_wide[["train", "val", "test"]]
    rec_wide = rec_wide[["train", "val", "test"]]
    seg_wide.columns = [f"{c}_segments" for c in seg_wide.columns]
    rec_wide.columns = [f"{c}_recordings" for c in rec_wide.columns]
    species_split_wide = seg_wide.join(rec_wide)
    species_split_wide["total_segments"] = species_split_wide[["train_segments", "val_segments", "test_segments"]].sum(axis=1)
    species_split_wide["total_recordings"] = species_split_wide[["train_recordings", "val_recordings", "test_recordings"]].sum(axis=1)
    species_split_wide.to_csv(outdir / "diagnostic_species_split_wide.csv")

    # 2. Species x split x recording quality.
    q = (
        meta.groupby(["species_label", "split", "recording_quality"], dropna=False)
        .agg(
            n_segments=("segment_id", "size"),
            n_recordings=("recording_id", "nunique"),
        )
        .reset_index()
    )
    q.to_csv(outdir / "diagnostic_species_split_quality.csv", index=False)

    # 3. Overall split composition, including segments and recordings.
    overall = (
        meta.groupby("split")
        .agg(
            n_segments=("segment_id", "size"),
            n_recordings=("recording_id", "nunique"),
            n_species=("species_label", "nunique"),
        )
        .reset_index()
    )
    overall["segment_fraction"] = overall["n_segments"] / overall["n_segments"].sum()
    overall.to_csv(outdir / "diagnostic_split_summary.csv", index=False)

    # 4. Species totals plus split coverage flags.
    totals = (
        meta.groupby("species_label")
        .agg(
            total_segments=("segment_id", "size"),
            total_recordings=("recording_id", "nunique"),
            qualities_present=("recording_quality", lambda x: ",".join(sorted(x.dropna().astype(str).unique()))),
        )
        .reset_index()
    )
    coverage = rec_wide.reset_index()
    totals = totals.merge(coverage, on="species_label", how="left")
    totals["n_splits"] = (totals[["train_recordings", "val_recordings", "test_recordings"]] > 0).sum(axis=1)
    totals["test_present"] = totals["test_recordings"] > 0
    totals["val_present"] = totals["val_recordings"] > 0
    totals.to_csv(outdir / "diagnostic_species_coverage.csv", index=False)

    # 5. Recording-level split integrity check.
    recording_split = meta.groupby("recording_id")["split"].nunique()
    leakage = recording_split[recording_split > 1].rename("n_splits").reset_index()
    leakage.to_csv(outdir / "diagnostic_recording_split_leakage.csv", index=False)

    # 6. Recording-level species purity/multiplicity. This identifies recordings
    # containing events from multiple collapsed species.
    rec_species = (
        meta.groupby("recording_id")
        .agg(
            split=("split", "first"),
            n_species=("species_label", "nunique"),
            species=("species_label", lambda x: ",".join(sorted(x.astype(str).unique()))),
            n_segments=("segment_id", "size"),
        )
        .reset_index()
    )
    rec_species["multi_species_recording"] = rec_species["n_species"] > 1
    rec_species.to_csv(outdir / "diagnostic_recording_species_composition.csv", index=False)

    # 7. Quality distribution by split.
    quality_split = (
        pd.crosstab(meta["split"], meta["recording_quality"], dropna=False)
        .reindex(index=["train", "val", "test"], fill_value=0)
    )
    quality_split.to_csv(outdir / "diagnostic_quality_by_split.csv")

    # 8. Compact human-readable report.
    lines = []
    lines.append("BirdNET species-level linear probe — dataset diagnostic report")
    lines.append("=" * 72)
    lines.append(f"Strong labeled segments: {len(meta)}")
    lines.append(f"Species: {meta['species_label'].nunique()}")
    lines.append(f"Recordings: {meta['recording_id'].nunique()}")
    lines.append("")
    lines.append("Segments and recordings by split:")
    lines.append(overall.to_string(index=False))
    lines.append("")
    lines.append("Species coverage:")
    lines.append(species_split_wide.to_string())
    lines.append("")
    missing_test = sorted(totals.loc[~totals["test_present"], "species_label"].tolist())
    missing_val = sorted(totals.loc[~totals["val_present"], "species_label"].tolist())
    lines.append(f"Species absent from TEST: {', '.join(missing_test) if missing_test else 'None'}")
    lines.append(f"Species absent from VALIDATION: {', '.join(missing_val) if missing_val else 'None'}")
    lines.append("")
    lines.append(f"Recordings appearing in >1 split: {len(leakage)}")
    lines.append(f"Multi-species recordings: {int(rec_species['multi_species_recording'].sum())}")
    lines.append("")
    lines.append("Species × split × quality:")
    lines.append(q.to_string(index=False))
    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- Segment counts and recording counts are reported separately; split integrity is evaluated at recording level.")
    lines.append("- Species absent from validation/test are not evaluated on those splits.")
    lines.append("- Quality is retained as a diagnostic variable and is not used as the classification target.")
    lines.append("- Multi-species recordings are flagged because recording-level grouping can still leave within-recording species mixtures.")
    (outdir / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nDiagnostic report generated:")
    print(f"  {outdir / 'diagnostic_report.txt'}")
    print(f"  {outdir / 'diagnostic_species_split_counts.csv'}")
    print(f"  {outdir / 'diagnostic_species_split_wide.csv'}")
    print(f"  {outdir / 'diagnostic_species_split_quality.csv'}")
    print(f"  {outdir / 'diagnostic_species_coverage.csv'}")
    print(f"  {outdir / 'diagnostic_recording_species_composition.csv'}")
    print(f"  {outdir / 'diagnostic_quality_by_split.csv'}")


def main():
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BirdNET 2.4 — recording-grouped species-level linear probe (Windows-safe)")
    print("=" * 78)

    X, meta = load_data(args.embedding_file, args.metadata)

    print(f"Usable strong-event embeddings: {len(meta)}")
    print(f"Embedding dimension: {X.shape[1]}")
    print(f"Classes: {meta.target.nunique()}")
    print(f"Recordings: {meta.recording_id.nunique()}")
    print("\nClass counts:")
    print(meta["target"].value_counts().sort_index().to_string())
    print("\nSpecies × recording-quality counts:")
    quality_counts = pd.crosstab(
        meta["species_label"],
        meta["recording_quality"],
        dropna=False,
    )
    print(quality_counts.to_string())
    quality_counts.to_csv(outdir / "species_by_quality_counts.csv")

    print("\nSplit counts:")
    print(meta["split"].value_counts().to_string())

    # ---------------------------------------------------------------
    # Dataset/split diagnostics. This does not affect model fitting.
    # ---------------------------------------------------------------
    generate_diagnostic_report(meta, outdir)

    # ---------------------------------------------------------------
    # Fixed train / validation / test split.
    # ---------------------------------------------------------------
    train_mask = meta["split"].eq("train").to_numpy()
    val_mask = meta["split"].eq("val").to_numpy()
    test_mask = meta["split"].eq("test").to_numpy()

    X_train, y_train = X[train_mask], meta.loc[train_mask, "target"].to_numpy()
    X_val, y_val = X[val_mask], meta.loc[val_mask, "target"].to_numpy()
    X_test, y_test = X[test_mask], meta.loc[test_mask, "target"].to_numpy()

    g_train = meta.loc[train_mask, "recording_id"].to_numpy()

    # ---------------------------------------------------------------
    # Hyperparameter selection happens using TRAIN only.
    # ---------------------------------------------------------------
    print("\nRunning recording-grouped CV on TRAIN...")

    log_cv = cv_score_logistic(
        X_train, y_train, g_train,
        args.logistic_c, args.cv_folds, args.random_state
    )

    knn_cv = cv_score_knn(
        X_train, y_train, g_train,
        args.knn_k, args.cv_folds, args.random_state
    )

    cv_results = pd.concat([log_cv, knn_cv], ignore_index=True)
    cv_results.to_csv(outdir / "grouped_cv_results.csv", index=False)

    print("\nGrouped CV results:")
    print(
        cv_results[
            ["model", "hyperparameter", "mean_macro_f1", "std_macro_f1"]
        ].to_string(index=False)
    )

    best_log = log_cv.iloc[log_cv["mean_macro_f1"].argmax()]
    best_knn = knn_cv.iloc[knn_cv["mean_macro_f1"].argmax()]

    # ---------------------------------------------------------------
    # Use validation set as an independent model-selection check.
    # We select the hyperparameter that performed best in TRAIN CV,
    # then report validation performance without using TEST.
    # ---------------------------------------------------------------
    best_c = float(best_log["value"])
    best_k = int(best_knn["value"])

    log_model = SoftmaxRegression(
        C=best_c,
        max_iter=1500,
        class_weight="balanced",
        random_state=args.random_state,
    )

    knn_model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=best_k,
            weights="distance",
            metric="euclidean",
        )),
    ])

    log_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    val_log = evaluate(log_model, X_val, y_val, "val_logistic", outdir)
    val_knn = evaluate(knn_model, X_val, y_val, "val_knn", outdir)

    print("\nValidation:")
    print(pd.DataFrame([val_log, val_knn]).to_string(index=False))

    # ---------------------------------------------------------------
    # Final models: fit train + validation, evaluate once on TEST.
    # ---------------------------------------------------------------
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_log = SoftmaxRegression(
        C=best_c,
        max_iter=1500,
        class_weight="balanced",
        random_state=args.random_state,
    )
    final_knn = clone(knn_model)

    final_log.fit(X_trainval, y_trainval)
    final_knn.fit(X_trainval, y_trainval)

    test_log = evaluate(final_log, X_test, y_test, "test_logistic", outdir)
    test_knn = evaluate(final_knn, X_test, y_test, "test_knn", outdir)

    test_results = pd.DataFrame([test_log, test_knn])
    test_results.to_csv(outdir / "test_metrics.csv", index=False)

    print("\nFINAL HELD-OUT TEST:")
    print(test_results.to_string(index=False))

    # ---------------------------------------------------------------
    # Per-class comparison for the two models.
    # ---------------------------------------------------------------
    log_report = pd.read_csv(
        outdir / "classification_report_test_logistic.csv",
        index_col=0
    )
    knn_report = pd.read_csv(
        outdir / "classification_report_test_knn.csv",
        index_col=0
    )

    common_classes = sorted(
        set(log_report.index) & set(knn_report.index)
        - {"accuracy", "macro avg", "weighted avg"}
    )

    if common_classes:
        comparison = pd.DataFrame({
            "class": common_classes,
            "logistic_precision": [
                log_report.loc[c, "precision"] for c in common_classes
            ],
            "logistic_recall": [
                log_report.loc[c, "recall"] for c in common_classes
            ],
            "logistic_f1": [
                log_report.loc[c, "f1-score"] for c in common_classes
            ],
            "knn_precision": [
                knn_report.loc[c, "precision"] for c in common_classes
            ],
            "knn_recall": [
                knn_report.loc[c, "recall"] for c in common_classes
            ],
            "knn_f1": [
                knn_report.loc[c, "f1-score"] for c in common_classes
            ],
        })
        comparison.to_csv(outdir / "per_class_test_comparison.csv", index=False)

    summary = {
        "embedding_file": str(args.embedding_file),
        "metadata_file": str(args.metadata),
        "embedding_dimension": int(X.shape[1]),
        "n_strong_labeled_segments": int(len(meta)),
        "n_classes": int(meta["target"].nunique()),
        "n_recordings": int(meta["recording_id"].nunique()),
        "train_segments": int(train_mask.sum()),
        "val_segments": int(val_mask.sum()),
        "test_segments": int(test_mask.sum()),
        "best_logistic_C": best_c,
        "best_knn_k": best_k,
        "logistic_train_cv_macro_f1_mean": float(best_log["mean_macro_f1"]),
        "logistic_train_cv_macro_f1_sd": float(best_log["std_macro_f1"]),
        "knn_train_cv_macro_f1_mean": float(best_knn["mean_macro_f1"]),
        "knn_train_cv_macro_f1_sd": float(best_knn["std_macro_f1"]),
        "logistic_validation": val_log,
        "knn_validation": val_knn,
        "logistic_test": test_log,
        "knn_test": test_knn,
    }

    with open(outdir / "linear_probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nOutputs written to:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
