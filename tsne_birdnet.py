#!/usr/bin/env python3
"""PCA + t-SNE visualization of BirdNET embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path,
                   default=Path("embeddings/birdnet_embeddings.npz"))
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--output-dir", type=Path,
                   default=Path("analysis/figures"))
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=("train", "val", "test"), default=None)
    p.add_argument("--l2-normalize", action="store_true")
    p.add_argument("--max-points", type=int, default=None)
    return p.parse_args()


def first_existing(df, names, what):
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"No {what} column found. Tried: {names}")


def species_label(df):
    for c in ("label", "labels", "species", "species_label", "class"):
        if c in df.columns:
            return (
                df[c].fillna("UNKNOWN").astype(str)
                .str.split(r"[;,|]").str[0].str.strip()
            )
    return pd.Series("UNKNOWN", index=df.index)


def plot_groups(df, column, filename, title, outdir):
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = df[column].fillna("UNKNOWN").astype(str)
    categories = labels.value_counts().index

    for cat in categories:
        m = labels.eq(cat)
        ax.scatter(df.loc[m, "tsne_1"], df.loc[m, "tsne_2"],
                   s=18, alpha=.7, label=f"{cat} (n={m.sum()})",
                   linewidths=0)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.grid(alpha=.15)
    if len(categories) <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    a = args()

    if not a.embeddings.exists():
        raise SystemExit(f"Embedding file not found: {a.embeddings}")

    data = np.load(a.embeddings)
    X = np.asarray(data["embeddings"], dtype=np.float32)

    meta = a.metadata or a.embeddings.with_name(
        a.embeddings.stem + "_metadata.csv"
    )
    if not meta.exists():
        raise SystemExit(f"Metadata file not found: {meta}")

    df = pd.read_csv(meta)
    if len(df) != len(X):
        raise SystemExit(
            f"Embedding/metadata mismatch: {len(X)} vs {len(df)}"
        )

    if a.split:
        c = first_existing(
            df, ("split", "dataset_split", "recording_split"), "split"
        )
        m = df[c].astype(str).str.lower().eq(a.split)
        X, df = X[m.to_numpy()], df.loc[m].copy()

    if a.max_points and len(X) > a.max_points:
        rng = np.random.default_rng(a.seed)
        ix = np.sort(rng.choice(len(X), a.max_points, replace=False))
        X, df = X[ix], df.iloc[ix].copy()

    if len(X) < 4:
        raise SystemExit("Need at least 4 samples.")
    if not 1 < a.perplexity < len(X):
        raise SystemExit(
            f"Perplexity must be >1 and < {len(X)}; got {a.perplexity}."
        )

    X0 = normalize(X) if a.l2_normalize else X
    n_pca = min(a.pca_components, X0.shape[0] - 1, X0.shape[1])
    pca = PCA(n_components=n_pca, random_state=a.seed)
    Xp = pca.fit_transform(X0)

    tsne = TSNE(
        n_components=2,
        perplexity=a.perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=a.iterations,
        random_state=a.seed,
    )
    Z = tsne.fit_transform(Xp)

    df = df.reset_index(drop=True)
    df["tsne_1"] = Z[:, 0]
    df["tsne_2"] = Z[:, 1]
    df["_species"] = species_label(df)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output_dir / "tsne_coordinates.csv", index=False)

    plot_groups(
        df, "_species", "tsne_species.png",
        "BirdNET embeddings — t-SNE by species", a.output_dir
    )

    for c, filename, title in (
        ("recording_id", "tsne_recording.png",
         "BirdNET embeddings — t-SNE by recording"),
        ("split", "tsne_split.png",
         "BirdNET embeddings — t-SNE by dataset split"),
    ):
        if c in df.columns:
            plot_groups(df, c, filename, title, a.output_dir)

    pd.DataFrame({
        "parameter": [
            "n_samples", "input_dimensions", "pca_dimensions",
            "pca_explained_variance", "perplexity", "iterations", "seed"
        ],
        "value": [
            len(X), X.shape[1], Xp.shape[1],
            pca.explained_variance_ratio_.sum(),
            a.perplexity, a.iterations, a.seed
        ],
    }).to_csv(a.output_dir / "tsne_parameters.csv", index=False)

    print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Saved: {a.output_dir / 'tsne_species.png'}")
    print(f"Saved: {a.output_dir / 'tsne_coordinates.csv'}")


if __name__ == "__main__":
    main()
