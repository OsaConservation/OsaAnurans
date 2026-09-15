#!/usr/bin/env python3
"""
Extract BirdNET embeddings for the segment files described by metadata.csv.

This script is written specifically for the metadata structure:

segment_id
recording_id
audio_file
segment_file
split
start_time
end_time
duration
source_type
labels
n_labels
event_start
event_end
event_label
n_events_in_window
weak_recording_labels

Expected project layout:

project/
├── dataset/
│   ├── metadata.csv
│   ├── strong/
│   │   └── *.wav
│   └── candidates/
│       └── *.wav
├── embeddings/
└── analysis/

The metadata contains paths such as:
    strong\\INCT4_...wav
    candidates\\INCT4_...wav

Paths are resolved relative to metadata.csv, so if metadata.csv is
dataset/metadata.csv, "strong\\file.wav" resolves to dataset/strong/file.wav.

By default ALL 2,956 rows are embedded. Use --source-type to restrict this.

BirdNET models:
    --model-version 2.4 --backend tf
    --model-version 3.0 --backend onnx

BirdNET 2.4 embeddings are 1024-D.
BirdNET 3.0 embeddings are 1280-D.

The official BirdNET Python API exposes acoustic model encoding through
model.encode(...). See:
https://github.com/birdnet-team/birdnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


REQUIRED_COLUMNS = {
    "segment_id",
    "recording_id",
    "segment_file",
    "split",
    "start_time",
    "end_time",
    "duration",
    "source_type",
    "labels",
    "event_label",
    "weak_recording_labels",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract BirdNET embeddings using the supplied metadata.csv."
    )

    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("dataset/metadata.csv"),
        help="Path to metadata.csv.",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("embeddings/birdnet_embeddings.npz"),
        help="Output embedding file.",
    )

    p.add_argument(
        "--model-version",
        choices=["2.4", "3.0"],
        default="2.4",
        help="BirdNET acoustic model version.",
    )

    p.add_argument(
        "--backend",
        default=None,
        help="BirdNET backend. Default: tf for 2.4, onnx for 3.0.",
    )

    p.add_argument(
        "--source-type",
        choices=["all", "strong_event", "weak_candidate"],
        default="all",
        help="Which metadata rows to embed.",
    )

    p.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Restrict extraction to a dataset split.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows, useful for testing.",
    )

    p.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Start at this zero-based metadata row after filtering.",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file.",
    )

    return p.parse_args()


def normalize_path(value: str) -> Path:
    """
    Normalize Windows-style metadata paths.

    The CSV contains e.g.
        candidates\\file.wav

    This makes those paths work whether the script is run on Windows,
    Linux, or WSL.
    """
    value = str(value).strip().strip('"').strip("'")
    value = value.replace("\\", "/")
    return Path(*value.split("/"))


def resolve_segment_path(segment_file: str, metadata_path: Path) -> Path:
    p = normalize_path(segment_file)

    if p.is_absolute() and p.exists():
        return p.resolve()

    # The segment_file values are relative to dataset/, i.e. the directory
    # containing metadata.csv.
    candidates = [
        metadata_path.parent / p,
        metadata_path.parent.parent / p,
        Path.cwd() / p,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return the most likely path for useful diagnostics.
    return (metadata_path.parent / p).resolve()


def load_birdnet(version: str, backend: str | None):
    try:
        import birdnet
    except ImportError as exc:
        raise SystemExit(
            "\nBirdNET is not installed.\n\n"
            "For BirdNET 2.4:\n"
            "    pip install \"birdnet[tf]\"\n\n"
            "For BirdNET 3.0 ONNX:\n"
            "    pip install birdnet\n"
        ) from exc

    if backend is None:
        backend = "tf" if version == "2.4" else "onnx"

    print(f"Loading BirdNET acoustic {version} ({backend})...")
    model = birdnet.load("acoustic", version, backend)
    return model, backend


def result_to_embedding(result) -> np.ndarray:
    """
    Convert BirdNET's encoding result into a single 1-D embedding.

    A 3-second input should produce one embedding. This function handles
    several result representations used by BirdNET versions and backends.
    """
    # Direct attributes commonly used by result objects.
    for attr in (
        "embeddings",
        "embedding",
        "encoding",
        "encodings",
    ):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if callable(value):
                continue
            try:
                arr = np.asarray(value, dtype=np.float32)
                if arr.size:
                    return reduce_embedding(arr)
            except Exception:
                pass

    # Dict-like results.
    if isinstance(result, dict):
        for key in ("embeddings", "embedding", "encoding", "encodings"):
            if key in result:
                arr = np.asarray(result[key], dtype=np.float32)
                return reduce_embedding(arr)

    # pandas/numpy-like result.
    for method in ("to_numpy", "numpy"):
        if hasattr(result, method):
            try:
                arr = np.asarray(getattr(result, method)(), dtype=np.float32)
                if arr.size:
                    return reduce_embedding(arr)
            except Exception:
                pass

    # Last resort: directly convert the returned object.
    try:
        arr = np.asarray(result, dtype=np.float32)
        if arr.size and arr.dtype.kind in "fc":
            return reduce_embedding(arr)
    except Exception:
        pass

    raise RuntimeError(
        "Could not locate an embedding in BirdNET's encode() result. "
        f"Returned type: {type(result)}"
    )


def reduce_embedding(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        # Expected shape for a single 3-s clip is normally (1, D).
        if arr.shape[0] == 1:
            return arr[0]

        # If the backend returns several frames for the clip, average them.
        return arr.mean(axis=0)

    if arr.ndim == 3:
        # Batch/time/embedding -> collapse batch and time.
        return arr.reshape(-1, arr.shape[-1]).mean(axis=0)

    raise ValueError(f"Unexpected BirdNET embedding shape: {arr.shape}")


def encode_one(model, path: Path) -> np.ndarray:
    """
    Encode one already-created 3-second segment.

    Important: we do NOT ask BirdNET to re-segment the original recording.
    Each row in metadata.csv maps to exactly one audio segment.
    """
    result = model.encode(str(path))
    embedding = result_to_embedding(result)

    if embedding.ndim != 1:
        raise ValueError(f"Embedding is not 1-D: {embedding.shape}")

    if embedding.size < 100:
        raise ValueError(
            f"Embedding dimension is unexpectedly small: {embedding.size}"
        )

    return embedding.astype(np.float32)


def validate_metadata(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "metadata.csv is missing required columns:\n"
            + "\n".join(sorted(missing))
        )


def main():
    args = parse_args()

    if not args.metadata.exists():
        raise SystemExit(f"Metadata file not found: {args.metadata}")

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"\nOutput already exists:\n  {args.output}\n\n"
            "Use --overwrite to replace it."
        )

    df = pd.read_csv(args.metadata)
    validate_metadata(df)

    print(f"Metadata rows: {len(df)}")

    # Filter by source type.
    if args.source_type != "all":
        df = df[
            df["source_type"].astype(str).str.lower()
            == args.source_type.lower()
        ].copy()

    # Filter by split.
    if args.split != "all":
        df = df[
            df["split"].astype(str).str.lower() == args.split.lower()
        ].copy()

    if args.start_row:
        df = df.iloc[args.start_row:].copy()

    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    if df.empty:
        raise SystemExit("No metadata rows remain after filtering.")

    print(f"Rows selected: {len(df)}")
    print("\nSource type:")
    print(df["source_type"].value_counts().to_string())

    print("\nSplit:")
    print(df["split"].value_counts().to_string())

    # Resolve all paths before loading BirdNET.
    paths = []
    missing = []

    for idx, row in df.iterrows():
        path = resolve_segment_path(row["segment_file"], args.metadata)
        paths.append((idx, path))

        if not path.exists():
            missing.append((idx, row["segment_file"], str(path)))

    if missing:
        print(
            f"\nERROR: {len(missing)} segment files were not found.",
            file=sys.stderr,
        )
        print("\nFirst missing files:", file=sys.stderr)
        for idx, stored, resolved in missing[:20]:
            print(
                f"  metadata row {idx}: {stored}\n"
                f"      looked for: {resolved}",
                file=sys.stderr,
            )

        raise SystemExit(
            "\nFix the dataset path before running extraction. "
            "The metadata itself was read correctly."
        )

    model, backend = load_birdnet(args.model_version, args.backend)

    embeddings = []
    successful_indices = []
    failures = []

    for idx, path in tqdm(
        paths,
        total=len(paths),
        desc=f"BirdNET {args.model_version} embeddings",
    ):
        try:
            emb = encode_one(model, path)

            embeddings.append(emb)
            successful_indices.append(idx)

        except Exception as exc:
            failures.append(
                {
                    "metadata_index": int(idx),
                    "segment_id": str(df.loc[idx, "segment_id"]),
                    "segment_file": str(df.loc[idx, "segment_file"]),
                    "error": repr(exc),
                }
            )

    if not embeddings:
        raise SystemExit(
            "BirdNET did not produce any embeddings. "
            "Check the BirdNET installation and model backend."
        )

    dimensions = sorted({int(e.shape[0]) for e in embeddings})

    if len(dimensions) != 1:
        raise SystemExit(
            f"Embeddings have inconsistent dimensions: {dimensions}"
        )

    X = np.vstack(embeddings).astype(np.float32)

    # Preserve exactly the metadata rows corresponding to X.
    used_metadata = df.loc[successful_indices].copy()
    used_metadata.insert(
        0,
        "embedding_row",
        np.arange(len(used_metadata)),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        embeddings=X,
        metadata_row_indices=np.asarray(
            successful_indices,
            dtype=np.int64,
        ),
    )

    metadata_output = args.output.with_name(
        args.output.stem + "_metadata.csv"
    )
    used_metadata.to_csv(metadata_output, index=False)

    report = {
        "metadata_file": str(args.metadata),
        "output_file": str(args.output),
        "model": f"acoustic-{args.model_version}",
        "backend": backend,
        "source_type": args.source_type,
        "split": args.split,
        "n_metadata_rows_selected": int(len(df)),
        "n_embeddings": int(len(X)),
        "embedding_dimension": int(X.shape[1]),
        "n_failures": int(len(failures)),
        "failures": failures,
    }

    report_output = args.output.with_name(
        args.output.stem + "_report.json"
    )
    report_output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nExtraction complete.")
    print(f"Embeddings:       {X.shape[0]} x {X.shape[1]}")
    print(f"Embedding file:   {args.output}")
    print(f"Metadata file:    {metadata_output}")
    print(f"Report:           {report_output}")

    if failures:
        print(
            f"\nWARNING: {len(failures)} segments failed. "
            "See the JSON report."
        )


if __name__ == "__main__":
    main()
