#!/usr/bin/env python3
"""
Research-grade BirdNET embedding extraction for a fixed-length anuran segment dataset.


Labels/targets are joined later by segment_id.

Tested design target:
    BirdNET 2.4
    backend: tf
    library: litert
    embedding dimension: 1024

The BirdNET package currently documents model.encode(...) as the embedding
interface for supported acoustic models. BirdNET 2.4 uses a 1024-dimensional
embedding and 3-second acoustic input. The script intentionally uses the
model.encode() API rather than prediction outputs.

Typical command on the user's dataset:

If segment audio is under a dedicated directory, replace --audio-dir accordingly.

Resume:
python extract_birdnet_embeddings.py ... --resume

Smoke test:
python extract_birdnet_embeddings.py ... --max-segments 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("extract_birdnet_embeddings")

REQUIRED_METADATA = (
    "segment_id",
    "recording_id",
    "site_id",
    "split",
    "source_type",
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
    ".aif",
    ".aiff",
    ".au",
    ".caf",
    ".mp3",
    ".opus",
    ".w64",
    ".rf64",
}

BIRDNET_VERSION = "2.4"
BIRDNET_BACKEND = "tf"
BIRDNET_LIBRARY = "litert"
EXPECTED_EMBEDDING_DIM = 1024


@dataclass
class Config:
    metadata: str
    audio_dir: str
    output_dir: str
    batch_size: int
    resume: bool
    overwrite: bool
    max_segments: int | None
    expected_duration: float | None
    expected_sample_rate: int | None
    normalize: bool
    save_every: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract BirdNET 2.4 embeddings from 3-second audio segments.",
    )

    p.add_argument("--metadata", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Checkpoint frequency. BirdNET 2.4 model.encode is called per file "
            "to preserve exact segment_id alignment; this value controls how "
            "often the resumable checkpoint is written."
        ),
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Write a checkpoint after this many newly processed segments.",
    )

    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete/replace the existing extraction after compatibility checks.",
    )

    p.add_argument("--max-segments", type=int, default=None)

    p.add_argument(
        "--expected-duration",
        type=float,
        default=3.0,
        help="Expected WAV duration in seconds. Use 0 to disable.",
    )

    p.add_argument(
        "--expected-sample-rate",
        type=int,
        default=None,
        help="Optional expected WAV sample rate. Use 0 to disable.",
    )

    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize each embedding before saving.",
    )

    a = p.parse_args()

    if a.batch_size < 1:
        p.error("--batch-size must be >= 1")
    if a.save_every < 1:
        p.error("--save-every must be >= 1")
    if a.max_segments is not None and a.max_segments < 1:
        p.error("--max-segments must be >= 1")

    return Config(
        metadata=a.metadata,
        audio_dir=a.audio_dir,
        output_dir=a.output_dir,
        batch_size=a.batch_size,
        resume=a.resume,
        overwrite=a.overwrite,
        max_segments=a.max_segments,
        expected_duration=None if a.expected_duration == 0 else a.expected_duration,
        expected_sample_rate=(
            None if a.expected_sample_rate in (None, 0) else a.expected_sample_rate
        ),
        normalize=a.normalize,
        save_every=a.save_every,
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_METADATA if c not in df.columns]
    if missing:
        raise ValueError(
            f"metadata.csv is missing required columns: {missing}"
        )

    out = df.copy()

    for col in REQUIRED_METADATA:
        out[col] = out[col].astype("string").fillna("").str.strip()

    if out["segment_id"].eq("").any():
        raise ValueError("metadata.csv contains blank segment_id values.")

    if out["recording_id"].eq("").any():
        raise ValueError("metadata.csv contains blank recording_id values.")

    if out["site_id"].eq("").any():
        raise ValueError("metadata.csv contains blank site_id values.")

    if out["split"].eq("").any():
        raise ValueError("metadata.csv contains blank split values.")

    if out["source_type"].eq("").any():
        raise ValueError("metadata.csv contains blank source_type values.")

    duplicated = out[out["segment_id"].duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated["segment_id"].drop_duplicates().head(10).tolist()
        raise ValueError(
            "segment_id must be unique. "
            f"Found {len(duplicated)} duplicate rows; examples={examples}"
        )

    return out.reset_index(drop=True)


def load_metadata(config: Config) -> pd.DataFrame:
    LOGGER.info("Reading metadata: %s", config.metadata)

    df = pd.read_csv(config.metadata)
    df = validate_metadata(df)

    if config.max_segments is not None:
        df = df.head(config.max_segments).copy()

    LOGGER.info(
        "Metadata: %d segments | %d recordings | %d sites",
        len(df),
        df["recording_id"].nunique(),
        df["site_id"].nunique(),
    )

    LOGGER.info(
        "Splits: %s",
        df["split"].value_counts(dropna=False).to_dict(),
    )

    LOGGER.info(
        "Source types: %s",
        df["source_type"].value_counts(dropna=False).to_dict(),
    )

    return df.reset_index(drop=True)


def build_audio_index(audio_dir: Path) -> dict[str, Path]:
    LOGGER.info("Indexing audio files: %s", audio_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(audio_dir)

    index: dict[str, Path] = {}
    duplicate_stems: dict[str, list[str]] = {}

    for path in audio_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        segment_id = path.stem

        if segment_id in index:
            duplicate_stems.setdefault(
                segment_id,
                [str(index[segment_id])],
            ).append(str(path))
        else:
            index[segment_id] = path

    if duplicate_stems:
        examples = list(duplicate_stems.items())[:10]
        raise RuntimeError(
            "Multiple audio files resolve to the same segment_id. "
            "This would make metadata/embedding alignment ambiguous. "
            f"Examples: {examples}"
        )

    LOGGER.info("Indexed %d audio files.", len(index))
    return index


def probe_wav(
    path: Path,
    expected_duration: float | None,
    expected_sample_rate: int | None,
) -> tuple[bool, str, float | None, int | None]:
    """
    Validate WAV container metadata without decoding the complete file.

    BirdNET itself performs the actual audio decoding/resampling.
    """
    if not path.exists():
        return False, "missing_audio_file", None, None

    if path.suffix.lower() != ".wav":
        # Let BirdNET/SoundFile handle supported non-WAV formats.
        return True, "", None, None

    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            sample_rate = wf.getframerate()
            duration = frames / sample_rate if sample_rate else None

        if expected_sample_rate is not None:
            if sample_rate != expected_sample_rate:
                return (
                    False,
                    f"sample_rate={sample_rate};expected={expected_sample_rate}",
                    duration,
                    sample_rate,
                )

        if expected_duration is not None and duration is not None:
            # Allow tiny container/rounding discrepancies.
            if abs(duration - expected_duration) > 0.08:
                return (
                    False,
                    f"duration={duration:.6f};expected={expected_duration:.6f}",
                    duration,
                    sample_rate,
                )

        return True, "", duration, sample_rate

    except Exception as exc:
        return (
            False,
            f"wav_probe_error:{type(exc).__name__}:{exc}",
            None,
            None,
        )


def import_birdnet():
    try:
        import birdnet
    except Exception as exc:
        raise RuntimeError(
            "Could not import the 'birdnet' package. "
            "Activate the same environment in which BirdNET 2.4 was tested."
        ) from exc

    return birdnet


def load_model(birdnet):
    LOGGER.info(
        "Loading BirdNET acoustic model %s | backend=%s | library=%s",
        BIRDNET_VERSION,
        BIRDNET_BACKEND,
        BIRDNET_LIBRARY,
    )

    # This is the model configuration already validated in the user's
    # environment. It also avoids the Windows app-data path problem by
    # respecting BIRDNET_APP_DATA if the user has set it.
    model = birdnet.load(
        "acoustic",
        BIRDNET_VERSION,
        BIRDNET_BACKEND,
        library=BIRDNET_LIBRARY,
    )

    return model


def result_to_embedding(result) -> np.ndarray:
    """
    Convert BirdNET AcousticDataEncodingResult to a single 1-D vector.

    For a 3-second segment with no overlap, BirdNET 2.4 should yield one
    embedding of dimension 1024.

    The user's previously observed result exposes:
        result.embeddings
        result.embeddings_masked
        result.emb_dim
        result.segment_duration_s
        result.overlap_duration_s
    """
    if not hasattr(result, "embeddings"):
        raise RuntimeError(
            "BirdNET encode result does not expose '.embeddings'. "
            f"Received type={type(result)!r}"
        )

    raw = result.embeddings

    # Some BirdNET versions expose a method rather than a materialized array
    # under certain result fields. Do not call arbitrary methods blindly.
    if callable(raw):
        raw = raw()

    arr = np.asarray(raw, dtype=np.float32)
    arr = np.squeeze(arr)

    # Expected shapes:
    #   (1, 1024) -> one 3-sec segment
    #   (1024,)   -> already flattened
    # If multiple temporal windows somehow appear, do not silently average
    # them: that would alter the representation definition.
    if arr.ndim == 1:
        embedding = arr
    elif arr.ndim == 2 and arr.shape[0] == 1:
        embedding = arr[0]
    else:
        raise RuntimeError(
            "Expected exactly one BirdNET embedding for one 3-second segment; "
            f"got shape={arr.shape}. Check segment duration/overlap."
        )

    if embedding.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape={embedding.shape}")

    if not np.all(np.isfinite(embedding)):
        raise RuntimeError("Embedding contains NaN or infinite values.")

    return embedding.astype(np.float32, copy=False)


def extract_one(model, audio_path: Path) -> np.ndarray:
    """
    Encode exactly one fixed segment.

    We intentionally use model.encode(), not model.predict(), because this
    stage produces learned feature representations rather than class scores.
    """
    result = model.encode(str(audio_path))
    embedding = result_to_embedding(result)

    if embedding.shape[0] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected BirdNET embedding dimension={embedding.shape[0]}; "
            f"expected={EXPECTED_EMBEDDING_DIM}."
        )

    return embedding


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a zero/non-finite embedding.")
    return x / norm


def configuration_dict(config: Config) -> dict:
    d = asdict(config)
    # These do not define the representation and may legitimately change
    # between runs.
    d.pop("resume", None)
    d.pop("overwrite", None)
    d.pop("max_segments", None)
    d.pop("save_every", None)
    return d


def manifest_compatible(path: Path, config: Config) -> bool:
    if not path.exists():
        return True

    manifest = json.loads(path.read_text(encoding="utf-8"))
    previous = manifest.get("configuration", {})

    current = configuration_dict(config)

    # String paths can be represented differently between invocations.
    previous.pop("metadata", None)
    current.pop("metadata", None)
    previous.pop("audio_dir", None)
    current.pop("audio_dir", None)
    previous.pop("output_dir", None)
    current.pop("output_dir", None)

    return previous == current


def load_checkpoint(
    output_dir: Path,
    config: Config,
    current_metadata: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Load a prior extraction only when its configuration matches.

    The checkpoint consists of:
        checkpoint_embeddings.npy
        checkpoint_metadata.csv
        extraction_manifest.json
    """
    manifest_path = output_dir / "extraction_manifest.json"
    matrix_path = output_dir / "checkpoint_embeddings.npy"
    metadata_path = output_dir / "checkpoint_metadata.csv"

    if not (
        manifest_path.exists()
        and matrix_path.exists()
        and metadata_path.exists()
    ):
        return {}

    if not manifest_compatible(manifest_path, config):
        if config.overwrite:
            LOGGER.warning(
                "Existing extraction is incompatible; --overwrite was supplied. "
                "Starting a new extraction."
            )
            return {}

        raise RuntimeError(
            "Existing extraction has incompatible model/extraction settings. "
            "Use --overwrite to start over."
        )

    old_meta = pd.read_csv(metadata_path)
    old_emb = np.load(matrix_path)

    if len(old_meta) != len(old_emb):
        raise RuntimeError(
            "Checkpoint metadata and embeddings have different row counts."
        )

    if "segment_id" not in old_meta.columns:
        raise RuntimeError("Checkpoint metadata lacks segment_id.")

    if old_meta["segment_id"].duplicated().any():
        raise RuntimeError("Checkpoint contains duplicate segment_id values.")

    allowed = set(current_metadata["segment_id"])

    state: dict[str, np.ndarray] = {}

    for i, sid in enumerate(old_meta["segment_id"].astype(str)):
        if sid not in allowed:
            continue

        vec = np.asarray(old_emb[i], dtype=np.float32)

        if vec.ndim != 1 or vec.shape[0] != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"Invalid checkpoint embedding for {sid}: shape={vec.shape}"
            )

        state[sid] = vec

    LOGGER.info(
        "Loaded %d completed embeddings from checkpoint.",
        len(state),
    )

    return state


def write_checkpoint(
    output_dir: Path,
    metadata: pd.DataFrame,
    embeddings_by_id: dict[str, np.ndarray],
    failures: list[dict],
    config: Config,
    embedding_dim: int | None,
) -> None:
    """
    Atomically write a complete checkpoint.

    The matrix and metadata are always written in identical segment_id order.
    """
    if embeddings_by_id:
        ids = [
            sid
            for sid in metadata["segment_id"].astype(str).tolist()
            if sid in embeddings_by_id
        ]

        matrix = np.vstack(
            [embeddings_by_id[sid] for sid in ids]
        ).astype(np.float32, copy=False)

        checkpoint_meta = metadata[
            metadata["segment_id"].astype(str).isin(set(ids))
        ].copy()

        order = {sid: i for i, sid in enumerate(ids)}
        checkpoint_meta["_order"] = (
            checkpoint_meta["segment_id"].astype(str).map(order)
        )
        checkpoint_meta = (
            checkpoint_meta.sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        if checkpoint_meta["segment_id"].astype(str).tolist() != ids:
            raise RuntimeError("Checkpoint alignment invariant failed.")

    else:
        matrix = np.empty(
            (0, EXPECTED_EMBEDDING_DIM if embedding_dim else 0),
            dtype=np.float32,
        )
        checkpoint_meta = metadata.iloc[0:0].copy()

    if len(checkpoint_meta) != matrix.shape[0]:
        raise RuntimeError(
            "Checkpoint metadata/embedding row count mismatch."
        )

    # Store only segment metadata needed to identify the embedding. No labels.
    keep = list(REQUIRED_METADATA) + ["audio_path"]
    checkpoint_meta = checkpoint_meta[keep]

    matrix_tmp = output_dir / "checkpoint_embeddings.npy.tmp"
    with matrix_tmp.open("wb") as f:
        np.save(f, matrix)
    matrix_tmp.replace(output_dir / "checkpoint_embeddings.npy")

    meta_tmp = output_dir / "checkpoint_metadata.csv.tmp"
    checkpoint_meta.to_csv(meta_tmp, index=False)
    meta_tmp.replace(output_dir / "checkpoint_metadata.csv")

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(
        output_dir / "embedding_failures.csv",
        index=False,
    )

    manifest = {
        "created_utc": utc_now(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "configuration": configuration_dict(config),
        "birdnet": {
            "package": "birdnet",
            "model_type": "acoustic",
            "version": BIRDNET_VERSION,
            "backend": BIRDNET_BACKEND,
            "library": BIRDNET_LIBRARY,
            "expected_embedding_dimension": EXPECTED_EMBEDDING_DIM,
            "normalized": bool(config.normalize),
        },
        "dataset": {
            "requested_segments": int(len(metadata)),
            "successful_segments": int(len(checkpoint_meta)),
            "failed_segments": int(len(failures)),
            "recordings": int(metadata["recording_id"].nunique()),
            "sites": int(metadata["site_id"].nunique()),
            "splits": metadata["split"].value_counts().to_dict(),
            "source_types": metadata["source_type"].value_counts().to_dict(),
        },
    }

    atomic_json_write(output_dir / "extraction_manifest.json", manifest)


def write_final_outputs(
    output_dir: Path,
    metadata: pd.DataFrame,
    embeddings_by_id: dict[str, np.ndarray],
    failures: list[dict],
    config: Config,
    elapsed_seconds: float,
) -> None:
    ids = [
        sid
        for sid in metadata["segment_id"].astype(str).tolist()
        if sid in embeddings_by_id
    ]

    if ids:
        matrix = np.vstack(
            [embeddings_by_id[sid] for sid in ids]
        ).astype(np.float32, copy=False)

        meta = metadata[
            metadata["segment_id"].astype(str).isin(set(ids))
        ].copy()

        order = {sid: i for i, sid in enumerate(ids)}
        meta["_order"] = meta["segment_id"].astype(str).map(order)
        meta = (
            meta.sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )
    else:
        matrix = np.empty(
            (0, EXPECTED_EMBEDDING_DIM),
            dtype=np.float32,
        )
        meta = metadata.iloc[0:0].copy()

    if len(meta) != matrix.shape[0]:
        raise RuntimeError("Final metadata/embedding alignment failed.")

    if matrix.size and not np.all(np.isfinite(matrix)):
        raise RuntimeError("Final embeddings contain non-finite values.")

    if len(meta) and meta["segment_id"].duplicated().any():
        raise RuntimeError("Final metadata contains duplicate segment_id.")

    keep = list(REQUIRED_METADATA) + ["audio_path"]
    meta = meta[keep]

    # The primary artifacts.
    matrix_tmp = output_dir / "embeddings.npy.tmp"
    with matrix_tmp.open("wb") as f:
        np.save(f, matrix)
    matrix_tmp.replace(output_dir / "embeddings.npy")

    meta_tmp = output_dir / "embedding_metadata.csv.tmp"
    meta.to_csv(meta_tmp, index=False)
    meta_tmp.replace(output_dir / "embedding_metadata.csv")

    pd.DataFrame(failures).to_csv(
        output_dir / "embedding_failures.csv",
        index=False,
    )

    summary = {
        "created_utc": utc_now(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "n_requested": int(len(metadata)),
        "n_success": int(len(meta)),
        "n_failed": int(len(failures)),
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "embedding_dimension": (
            int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else None
        ),
        "n_recordings": int(metadata["recording_id"].nunique()),
        "n_sites": int(metadata["site_id"].nunique()),
        "split_counts": metadata["split"].value_counts().to_dict(),
        "source_type_counts": metadata["source_type"].value_counts().to_dict(),
        "success_split_counts": (
            meta["split"].value_counts().to_dict()
            if len(meta)
            else {}
        ),
        "success_source_type_counts": (
            meta["source_type"].value_counts().to_dict()
            if len(meta)
            else {}
        ),
        "model": {
            "package": "birdnet",
            "model": "acoustic",
            "version": BIRDNET_VERSION,
            "backend": BIRDNET_BACKEND,
            "library": BIRDNET_LIBRARY,
            "embedding_dimension": EXPECTED_EMBEDDING_DIM,
            "l2_normalized": bool(config.normalize),
        },
    }

    atomic_json_write(output_dir / "extraction_summary.json", summary)


def main() -> int:
    config = parse_args()
    setup_logging()

    metadata_path = Path(config.metadata)
    audio_dir = Path(config.audio_dir)
    output_dir = Path(config.output_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if not audio_dir.exists():
        raise FileNotFoundError(audio_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(config)

    # Build the segment_id -> audio path mapping.
    audio_index = build_audio_index(audio_dir)

    metadata["audio_path"] = metadata["segment_id"].map(
        lambda sid: str(audio_index.get(sid, ""))
    )

    # Preflight audio checks.
    probe_records = []
    for row in metadata.itertuples(index=False):
        path = Path(row.audio_path) if row.audio_path else Path("")

        ok, reason, duration, sample_rate = probe_wav(
            path,
            config.expected_duration,
            config.expected_sample_rate,
        )

        probe_records.append(
            {
                "segment_id": row.segment_id,
                "audio_probe_ok": ok,
                "audio_probe_reason": reason,
                "audio_duration_seconds": duration,
                "audio_sample_rate": sample_rate,
            }
        )

    probe = pd.DataFrame(probe_records)

    metadata = metadata.merge(
        probe,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    preflight_failures = metadata[
        ~metadata["audio_probe_ok"].fillna(False)
    ].copy()

    if len(preflight_failures):
        LOGGER.warning(
            "%d segments failed audio preflight.",
            len(preflight_failures),
        )

    metadata_ok = metadata[
        metadata["audio_probe_ok"].fillna(False)
    ].copy()

    if metadata_ok.empty:
        raise RuntimeError("No segments passed audio preflight.")

    # Resume only from a compatible representation.
    embeddings_by_id: dict[str, np.ndarray] = {}

    if config.resume:
        embeddings_by_id = load_checkpoint(
            output_dir,
            config,
            metadata_ok,
        )

    todo = metadata_ok[
        ~metadata_ok["segment_id"].astype(str).isin(embeddings_by_id)
    ].copy()

    LOGGER.info(
        "Extraction plan: requested=%d | already_done=%d | to_process=%d | "
        "preflight_failures=%d",
        len(metadata),
        len(embeddings_by_id),
        len(todo),
        len(preflight_failures),
    )

    birdnet = import_birdnet()
    model = load_model(birdnet)

    failures: list[dict] = []

    # Carry preflight failures into the permanent audit file.
    for row in preflight_failures.itertuples(index=False):
        failures.append(
            {
                "segment_id": row.segment_id,
                "recording_id": row.recording_id,
                "site_id": row.site_id,
                "split": row.split,
                "source_type": row.source_type,
                "audio_path": row.audio_path,
                "failure_stage": "preflight",
                "failure_type": "audio_validation",
                "failure_message": row.audio_probe_reason,
            }
        )

    started = time.time()
    processed_since_checkpoint = 0

    for i, row in enumerate(todo.itertuples(index=False), start=1):
        sid = str(row.segment_id)
        audio_path = Path(row.audio_path)

        try:
            vector = extract_one(model, audio_path)

            if config.normalize:
                vector = l2_normalize(vector).astype(np.float32)

            embeddings_by_id[sid] = vector
            processed_since_checkpoint += 1

        except Exception as exc:
            LOGGER.error(
                "FAILED %s | %s | %s",
                sid,
                type(exc).__name__,
                exc,
            )

            failures.append(
                {
                    "segment_id": sid,
                    "recording_id": row.recording_id,
                    "site_id": row.site_id,
                    "split": row.split,
                    "source_type": row.source_type,
                    "audio_path": row.audio_path,
                    "failure_stage": "embedding",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
            )

        if (
            processed_since_checkpoint >= config.save_every
            or i == len(todo)
        ):
            write_checkpoint(
                output_dir=output_dir,
                metadata=metadata,
                embeddings_by_id=embeddings_by_id,
                failures=failures,
                config=config,
                embedding_dim=EXPECTED_EMBEDDING_DIM,
            )
            processed_since_checkpoint = 0

        if i == 1 or i % 100 == 0 or i == len(todo):
            LOGGER.info(
                "Progress %d/%d | successful=%d | failures=%d | %.1f%%",
                i,
                len(todo),
                len(embeddings_by_id),
                len(failures),
                100.0 * i / len(todo),
            )

    elapsed = time.time() - started

    # Remove temporary probe-only columns before final output.
    final_metadata = metadata[
        list(REQUIRED_METADATA) + ["audio_path"]
    ].copy()

    write_final_outputs(
        output_dir=output_dir,
        metadata=final_metadata,
        embeddings_by_id=embeddings_by_id,
        failures=failures,
        config=config,
        elapsed_seconds=elapsed,
    )

    # Post-write integrity audit.
    saved_meta = pd.read_csv(output_dir / "embedding_metadata.csv")
    saved_emb = np.load(output_dir / "embeddings.npy")

    if saved_emb.ndim != 2:
        raise RuntimeError(
            f"embeddings.npy must be 2-D; got {saved_emb.shape}"
        )

    if saved_emb.shape[0] != len(saved_meta):
        raise RuntimeError(
            "POST-WRITE FAILURE: metadata rows != embedding rows."
        )

    if saved_emb.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            "POST-WRITE FAILURE: unexpected embedding dimension "
            f"{saved_emb.shape[1]}."
        )

    if len(saved_meta) and saved_meta["segment_id"].duplicated().any():
        raise RuntimeError(
            "POST-WRITE FAILURE: duplicate segment_id."
        )

    if saved_emb.size and not np.all(np.isfinite(saved_emb)):
        raise RuntimeError(
            "POST-WRITE FAILURE: non-finite embedding values."
        )

    LOGGER.info("=" * 72)
    LOGGER.info("Embedding extraction complete.")
    LOGGER.info("Successful: %d / %d", len(saved_meta), len(metadata))
    LOGGER.info("Failures:   %d", len(failures))
    LOGGER.info("Shape:      %s", saved_emb.shape)
    LOGGER.info("Output:     %s", output_dir.resolve())
    LOGGER.info("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
