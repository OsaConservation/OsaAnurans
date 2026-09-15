#!/usr/bin/env python
"""
Publication-quality visualization of BirdNET 2.4 embeddings.

Pipeline:
    1024-D BirdNET embeddings
        -> z-score features
        -> PCA (50 dimensions)
        -> t-SNE (2 dimensions)

Outputs:
    pca_coordinates.csv
    pca_variance.csv
    tsne_coordinates.csv
    figures/
        pca_2d_species.png / .pdf
        tsne_2d_species.png / .pdf
        tsne_2d_split.png / .pdf
        tsne_2d_source_type.png / .pdf
        pca_explained_variance.png / .pdf

Important:
    `weak_recording_labels` are NOT used as segment-level class labels.
    Only strong-event `event_label` values are used for the species plot.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    p = argparse.ArgumentParser(
        description="PCA/t-SNE visualization of BirdNET 2.4 embeddings."
    )
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max-iter", type=int, default=1500)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def species_label(row):
    """
    Use only confirmed strong-event labels.

    Weak candidate segments are deliberately treated as Unlabeled.
    """
    source = str(row.get("source_type", ""))

    event = row.get("event_label", np.nan)

    if (
        source == "strong_event"
        and pd.notna(event)
        and str(event).strip()
        and str(event).lower() not in {"nan", "none"}
    ):
        return str(event).strip()

    return "Unlabeled"


def configure_matplotlib():
    """
    Conservative publication settings.

    Figures are exported at 600 dpi PNG plus vector PDF.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def save_figure(fig, path):
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_categorical_scatter(
    x,
    y,
    groups,
    title,
    xlabel,
    ylabel,
    output_path,
    point_size=13,
    alpha=0.65,
):
    """
    Publication-style categorical scatter plot.

    The matplotlib default qualitative palette is used rather than hard-coding
    colors, allowing the script to remain neutral and easy to modify.
    """
    groups = pd.Series(groups).astype(str).to_numpy()
    unique = sorted(np.unique(groups))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    cmap = plt.get_cmap("tab20")

    for i, group in enumerate(unique):
        mask = groups == group

        if group == "Unlabeled":
            ax.scatter(
                x[mask],
                y[mask],
                s=point_size,
                marker=".",
                alpha=0.22,
                rasterized=True,
                label=group,
            )
        else:
            ax.scatter(
                x[mask],
                y[mask],
                s=point_size,
                marker="o",
                alpha=alpha,
                rasterized=True,
                label=group,
            )

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(
        True,
        alpha=0.15,
        linewidth=0.5,
        linestyle="-",
    )

    ax.legend(
        title="Group",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        markerscale=1.5,
        borderaxespad=0,
    )

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_pca_variance(pca, output_path):
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    individual = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.plot(
        components,
        cumulative,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Cumulative",
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("BirdNET embedding variance explained", pad=10)
    ax.set_xlim(1, len(components))
    ax.set_ylim(0, 100)

    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, output_path)

    # Save individual variance as well in the CSV; the figure emphasizes
    # cumulative variance because it is most useful for selecting PCA size.


def main():
    args = parse_args()

    output = Path(args.output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    print("=" * 72)
    print("BirdNET 2.4 embedding visualization")
    print("=" * 72)

    X = np.load(args.embedding_file)
    metadata = pd.read_csv(args.metadata)

    print(f"Embeddings : {X.shape}")
    print(f"Metadata   : {metadata.shape}")

    if X.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got {X.shape}")

    if X.shape[1] != 1024:
        raise ValueError(
            f"Expected BirdNET 2.4 embeddings with 1024 dimensions; "
            f"got {X.shape[1]}"
        )

    if len(X) != len(metadata):
        raise ValueError(
            f"Embedding rows ({len(X)}) do not match metadata rows "
            f"({len(metadata)})"
        )

    if "segment_id" not in metadata.columns:
        raise ValueError("Metadata must contain segment_id.")

    # Respect extraction status if present.
    if "embedding_status" in metadata.columns:
        keep = metadata["embedding_status"].astype(str).eq("success").to_numpy()

        if not np.all(keep):
            print(
                f"Skipping {(~keep).sum()} unsuccessful embeddings; "
                f"using {keep.sum()}."
            )

        X = X[keep]
        metadata = metadata.loc[keep].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------
    #
    # Each BirdNET dimension is standardized before PCA. This prevents
    # dimensions with larger raw variance from dominating the PCA.
    #
    X = X.astype(np.float32, copy=False)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Xz = (X - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    n_pca = min(
        args.pca_components,
        Xz.shape[0] - 1,
        Xz.shape[1],
    )

    print(f"PCA dimensions: {n_pca}")

    pca = PCA(
        n_components=n_pca,
        svd_solver="auto",
        random_state=args.random_state,
    )

    X_pca = pca.fit_transform(Xz)

    pca_coordinates = pd.DataFrame(
        X_pca,
        columns=[f"PC{i + 1}" for i in range(n_pca)],
    )

    pca_coordinates.insert(
        0,
        "segment_id",
        metadata["segment_id"].to_numpy(),
    )

    pca_coordinates.to_csv(
        output / "pca_coordinates.csv",
        index=False,
    )

    variance = pd.DataFrame({
        "component": np.arange(1, n_pca + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_percent": (
            pca.explained_variance_ratio_ * 100
        ),
        "cumulative_explained_variance_percent": (
            np.cumsum(pca.explained_variance_ratio_) * 100
        ),
    })

    variance.to_csv(
        output / "pca_variance.csv",
        index=False,
    )

    plot_pca_variance(
        pca,
        figures / "pca_explained_variance",
    )

    # Species labels for visualization.
    species = metadata.apply(species_label, axis=1)

    # PCA 2-D.
    plot_categorical_scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        species,
        "BirdNET embedding space: PCA",
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
        figures / "pca_2d_species",
    )

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    #
    # t-SNE is run on PCA-reduced features rather than the original 1024-D
    # representation. This is a standard dimensionality-reduction strategy
    # for visualization and substantially reduces computation/noise.
    #
    perplexity = args.perplexity

    # sklearn requires perplexity < n_samples. Keep the requested value
    # whenever possible.
    perplexity = min(
        perplexity,
        max(5.0, (len(X_pca) - 1) / 3.0),
    )

    print(f"t-SNE perplexity: {perplexity:.1f}")
    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=args.max_iter,
        random_state=args.random_state,
        metric="euclidean",
        verbose=1,
    )

    X_tsne = tsne.fit_transform(X_pca)

    tsne_coordinates = metadata.copy()
    tsne_coordinates["species_for_plot"] = species.to_numpy()
    tsne_coordinates["tsne_1"] = X_tsne[:, 0]
    tsne_coordinates["tsne_2"] = X_tsne[:, 1]

    tsne_coordinates.to_csv(
        output / "tsne_coordinates.csv",
        index=False,
    )

    # Main species visualization.
    plot_categorical_scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        species,
        "BirdNET embedding space: t-SNE",
        "t-SNE 1",
        "t-SNE 2",
        figures / "tsne_2d_species",
        point_size=14,
        alpha=0.68,
    )

    # Split diagnostic.
    if "split" in metadata.columns:
        plot_categorical_scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            metadata["split"],
            "BirdNET embedding space: t-SNE by data split",
            "t-SNE 1",
            "t-SNE 2",
            figures / "tsne_2d_split",
            point_size=14,
            alpha=0.65,
        )

    # Strong-event versus weak-candidate diagnostic.
    if "source_type" in metadata.columns:
        plot_categorical_scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            metadata["source_type"],
            "BirdNET embedding space: t-SNE by annotation source",
            "t-SNE 1",
            "t-SNE 2",
            figures / "tsne_2d_source_type",
            point_size=14,
            alpha=0.65,
        )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"Embeddings used : {len(X)}")
    print(f"PCA dimensions  : {n_pca}")
    print(f"t-SNE perplexity: {perplexity:.1f}")
    print(f"Output          : {output}")
    print()
    print("Figures:")
    for f in sorted(figures.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
