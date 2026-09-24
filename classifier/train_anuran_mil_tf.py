#!/usr/bin/env python
"""
TensorFlow/Keras MIL classifier for the AnuraSet BirdNET embeddings.

This version fixes the previous MIL data-flow error:

IMPORTANT:
    MIL bags contain ALL 3-second segments belonging to a recording,
    including both:
        source_type == "strong_event"
        source_type == "weak_candidate"

Strong segment supervision is restricted to confirmed strong positives:
    target == 1

Strong target NaN is UNKNOWN and is never converted to a negative.

Weak labels remain recording-level:
    presence == 1 -> recording positive
    presence == 0 -> recording negative evidence

The four-site LOSO protocol is performed by site_id. Validation is split
by recording_id inside the three training sites. The held-out site is never
used for fitting or threshold selection.

Model:
    1024-D BirdNET embedding
        -> shared segment encoder
        -> segment species logits
        -> species-specific attention
        -> attention MIL recording logits
        -> 42-species multilabel prediction

The recording MIL prediction is tied to the segment classifier rather than
using an unrelated recording-only classifier. This makes weak supervision
actually train the segment-level representation.

Primary evaluation:
    recording-level metrics on the independently supplied weak labels.

Secondary diagnostic:
    segment-level positive-vs-unlabeled proxy metrics on strong_event test
    segments. These are explicitly diagnostic because NaN strong labels are
    unknown rather than confirmed negatives.

Threshold diagnostics:
    validation_positive_recordings, selected_threshold, validation_f1,
    test_positive_recordings, test_f1, test_average_precision, test_roc_auc.
    Validation thresholds are selected only on training-site validation
    recordings. Species with zero held-out positives are retained in the
    detailed table with NaN test metrics and status=no_test_positives.

Dependencies:
    numpy
    pandas
    tensorflow
    scikit-learn

Example:
    python train_anuran_mil_tf.py

If weak files are elsewhere:
    python train_anuran_mil_tf.py ^
      --targets "D:\\path\\strong_targets(3).csv" ^
      --weak-long "D:\\path\\recording_weak_labels_long(3).csv" ^
      --weak-clean "D:\\path\\recording_weak_labels_clean(1).csv"
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
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
DEFAULT_WEAK_LONG = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\targets\recording_weak_labels_long.csv"
)
DEFAULT_WEAK_CLEAN = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\targets\recording_weak_labels_clean.csv"
)
DEFAULT_OUTPUT = Path(
    r"D:\Acoustics\AnuraSet_3sec_all\classifier_results_mil_tf"
)

SEED = 42
MIN_STANDARD_SUPPORT = 50


def parse_args():
    p = argparse.ArgumentParser(
        description="TensorFlow/Keras weakly supervised attention-MIL anuran classifier."
    )

    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--embedding-metadata", type=Path, default=DEFAULT_EMBED_METADATA)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    p.add_argument("--weak-long", type=Path, default=DEFAULT_WEAK_LONG)
    p.add_argument("--weak-clean", type=Path, default=DEFAULT_WEAK_CLEAN)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--attention-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--weak-loss-weight", type=float, default=1.0)
    p.add_argument("--strong-loss-weight", type=float, default=1.0)
    p.add_argument("--consistency-weight", type=float, default=0.25)
    p.add_argument("--early-stopping-patience", type=int, default=8)

    p.add_argument(
        "--max-segments-per-recording",
        type=int,
        default=20,
        help="Maximum segments retained per one-minute recording. "
             "Your 3-sec non-overlapping recordings normally contain <=20.",
    )

    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="TensorFlow intra/inter-op threads. 0 leaves TensorFlow defaults.",
    )

    return p.parse_args()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow(args):
    if args.threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    gpus = tf.config.list_physical_devices("GPU")

    print(f"TensorFlow version: {tf.__version__}")

    if gpus:
        print(f"TensorFlow GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print(
            "TensorFlow GPU not detected; training will use CPU."
        )
        print(
            "Intel Iris Xe is not automatically used by standard TensorFlow "
            "Windows installations."
        )


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name} missing required columns: {missing}"
        )


def load_weak_labels(args, species):
    """
    Prefer the long weak-label table.

    Expected long columns:
        recording_id
        species
        presence

    If only the clean wide table is available, SPECIES_* columns are
    converted to the same long representation.
    """
    if args.weak_long.exists():
        weak = pd.read_csv(args.weak_long)

        require_columns(
            weak,
            ["recording_id", "species", "presence"],
            "weak long labels",
        )

        weak = weak[
            ["recording_id", "species", "presence"]
        ].copy()

        weak["recording_id"] = (
            weak["recording_id"].astype(str)
        )
        weak["species"] = weak["species"].astype(str)

        weak["presence"] = pd.to_numeric(
            weak["presence"],
            errors="coerce",
        )

        weak = weak[
            weak["species"].isin(species)
        ].copy()

        weak["presence"] = (
            weak["presence"].fillna(0).gt(0).astype(float)
        )

    elif args.weak_clean.exists():
        wide = pd.read_csv(args.weak_clean)

        require_columns(
            wide,
            ["recording_id"],
            "weak clean labels",
        )

        rows = []

        for sp in species:
            col = f"SPECIES_{sp}"

            if col not in wide.columns:
                continue

            tmp = wide[
                ["recording_id", col]
            ].copy()

            tmp["species"] = sp

            tmp["presence"] = (
                pd.to_numeric(
                    tmp[col],
                    errors="coerce",
                )
                .fillna(0)
                .gt(0)
                .astype(float)
            )

            rows.append(
                tmp[
                    ["recording_id", "species", "presence"]
                ]
            )

        weak = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                columns=[
                    "recording_id",
                    "species",
                    "presence",
                ]
            )
        )

    else:
        print(
            "WARNING: no weak-label file found. "
            "The model will only have strong positive supervision."
        )

        weak = pd.DataFrame(
            columns=[
                "recording_id",
                "species",
                "presence",
            ]
        )

    weak["recording_id"] = (
        weak["recording_id"].astype(str)
    )

    weak["species"] = (
        weak["species"].astype(str)
    )

    if weak.duplicated(
        ["recording_id", "species"]
    ).any():
        print(
            "WARNING: duplicate recording/species weak labels found. "
            "Reducing duplicates by maximum presence."
        )

        weak = (
            weak.groupby(
                ["recording_id", "species"],
                as_index=False,
            )["presence"]
            .max()
        )

    return weak


def load_data(args):
    print("=" * 80)
    print("TENSORFLOW/KERAS ATTENTION-MIL ANURAN CLASSIFIER")
    print("=" * 80)

    X = np.load(
        args.embeddings,
        mmap_mode="r",
    )

    embedding_metadata = pd.read_csv(
        args.embedding_metadata
    )

    metadata = pd.read_csv(
        args.metadata
    )

    targets = pd.read_csv(
        args.targets
    )

    require_columns(
        embedding_metadata,
        ["segment_id"],
        "embedding metadata",
    )

    require_columns(
        metadata,
        [
            "segment_id",
            "recording_id",
            "site_id",
            "split",
            "source_type",
        ],
        "metadata",
    )

    require_columns(
        targets,
        [
            "segment_id",
            "recording_id",
            "site_id",
            "split",
            "source_type",
        ],
        "targets",
    )

    if X.ndim != 2 or X.shape[1] != 1024:
        raise RuntimeError(
            f"Expected embeddings shape (N,1024), got {X.shape}"
        )

    if len(X) != len(embedding_metadata):
        raise RuntimeError(
            "Embedding count does not match embedding metadata."
        )

    if len(X) != len(metadata):
        raise RuntimeError(
            "Embedding count does not match metadata."
        )

    if len(X) != len(targets):
        raise RuntimeError(
            "Embedding count does not match targets."
        )

    segment_ids = metadata[
        "segment_id"
    ].astype(str).to_numpy()

    emb_segment_ids = embedding_metadata[
        "segment_id"
    ].astype(str).to_numpy()

    target_segment_ids = targets[
        "segment_id"
    ].astype(str).to_numpy()

    if not np.array_equal(
        segment_ids,
        emb_segment_ids,
    ):
        raise RuntimeError(
            "Embedding metadata segment_id order does not match metadata."
        )

    if not np.array_equal(
        segment_ids,
        target_segment_ids,
    ):
        raise RuntimeError(
            "Targets segment_id order does not match metadata."
        )

    for col in [
        "recording_id",
        "site_id",
        "split",
        "source_type",
    ]:
        if not np.array_equal(
            metadata[col].astype(str).to_numpy(),
            targets[col].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"targets[{col}] does not exactly match metadata[{col}]."
            )

    id_columns = {
        "segment_id",
        "recording_id",
        "site_id",
        "split",
        "source_type",
    }

    species = [
        c for c in targets.columns
        if c not in id_columns
    ]

    for sp in species:
        numeric = pd.to_numeric(
            targets[sp],
            errors="coerce",
        )

        invalid = (
            targets[sp].notna()
            & numeric.isna()
        )

        if invalid.any():
            raise RuntimeError(
                f"Non-numeric target values found in {sp}."
            )

        invalid = (
            numeric.notna()
            & ~numeric.isin([0, 1])
        )

        if invalid.any():
            raise RuntimeError(
                f"Invalid target values in {sp}: "
                f"{numeric[invalid].unique()[:10]}"
            )

    weak = load_weak_labels(
        args,
        species,
    )

    metadata_recordings = set(
        metadata["recording_id"]
        .astype(str)
    )

    weak_recordings = set(
        weak["recording_id"]
        .astype(str)
    )

    missing_weak_recordings = (
        weak_recordings
        - metadata_recordings
    )

    print(f"Embeddings: {X.shape}")
    print(f"Segments:   {len(metadata):,}")
    print(f"Species:    {len(species)}")
    print(
        "Sites:      "
        + str(
            sorted(
                metadata["site_id"]
                .astype(str)
                .unique()
            )
        )
    )
    print(
        f"Weak recordings: {len(weak_recordings):,}"
    )

    if missing_weak_recordings:
        print(
            "WARNING: "
            f"{len(missing_weak_recordings)} weak recordings "
            "are absent from metadata."
        )

    print()
    print("SOURCE DISTRIBUTION:")
    print(
        metadata["source_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("LABEL SEMANTICS:")
    print("  strong target 1   = confirmed segment positive")
    print("  strong target NaN = unknown, NOT absence")
    print("  weak presence 1   = species present somewhere in recording")
    print("  weak presence 0   = recording-level absence evidence")
    print()
    print(
        "MIL bags will contain ALL segments from each recording, "
        "including weak_candidate segments."
    )

    return (
        X,
        embedding_metadata,
        metadata,
        targets,
        species,
        weak,
    )


def build_recording_index(metadata):
    """
    recording_id -> ordered global segment indices.
    """
    recording_to_indices = {}

    for idx, recording_id in enumerate(
        metadata["recording_id"].astype(str)
    ):
        recording_to_indices.setdefault(
            recording_id,
            [],
        ).append(idx)

    return recording_to_indices


def build_bags(
    recording_ids,
    recording_to_indices,
    metadata,
    targets,
    weak_lookup,
    species,
    max_segments,
    seed,
):
    """
    Build MIL bags from ALL segments in each selected recording.

    Critical correction relative to the previous script:
        We do NOT filter by source_type before making the bag.

    If a recording has both strong_event and weak_candidate segments,
    both are present in the bag.

    If a recording has more than max_segments:
        1. retain all confirmed strong-positive segments where possible
        2. fill remaining slots randomly from the other segments

    Strong supervision is handled later by the loss function and only uses
    target == 1.
    """
    rng = np.random.default_rng(seed)

    bags = []

    for recording_id in recording_ids:
        recording_id = str(recording_id)

        all_indices = np.asarray(
            recording_to_indices[
                recording_id
            ],
            dtype=np.int64,
        )

        selected = all_indices.copy()

        if (
            max_segments is not None
            and len(selected) > max_segments
        ):
            # Identify segments with at least one confirmed strong positive.
            target_values = targets.iloc[
                selected
            ][species].to_numpy(
                dtype=np.float32
            )

            positive_rows = np.isfinite(
                target_values
            ) & (
                target_values == 1
            )

            positive_indices = selected[
                positive_rows.any(axis=1)
            ]

            positive_indices = list(
                dict.fromkeys(
                    positive_indices.tolist()
                )
            )

            if len(positive_indices) >= max_segments:
                selected = rng.choice(
                    positive_indices,
                    size=max_segments,
                    replace=False,
                )
            else:
                selected_list = list(
                    positive_indices
                )

                remaining = [
                    int(i)
                    for i in selected
                    if int(i)
                    not in set(selected_list)
                ]

                n_extra = (
                    max_segments
                    - len(selected_list)
                )

                if n_extra > 0 and remaining:
                    extra = rng.choice(
                        remaining,
                        size=min(
                            n_extra,
                            len(remaining),
                        ),
                        replace=False,
                    ).tolist()

                    selected_list.extend(extra)

                selected = np.asarray(
                    selected_list,
                    dtype=np.int64,
                )

        # Weak labels are recording-level. A weak absence (0) is negative
        # evidence only when there is no independent strong positive for
        # that species anywhere in the same recording. If the sources
        # conflict, mask the weak-negative cell instead of forcing the
        # model to satisfy contradictory supervision.
        #
        # Inspect ALL segments in the recording, not only the selected MIL
        # bag, so max_segments cannot hide a confirmed strong positive.
        all_target_values = targets.iloc[
            all_indices
        ][species].to_numpy(
            dtype=np.float32
        )

        strong_positive_by_species = (
            np.isfinite(all_target_values)
            & (all_target_values == 1)
        ).any(axis=0)

        weak_y = np.full(
            len(species),
            np.nan,
            dtype=np.float32,
        )

        for j, sp in enumerate(species):
            value = weak_lookup.get(
                (recording_id, sp)
            )

            if value is None:
                continue

            value = float(value)

            if value == 0.0 and strong_positive_by_species[j]:
                # Weak-negative / strong-positive conflict: do not apply
                # weak loss for this recording/species cell.
                continue

            weak_y[j] = value

        bags.append(
            {
                "recording_id": recording_id,
                "indices": np.asarray(
                    selected,
                    dtype=np.int64,
                ),
                "weak_y": weak_y,
            }
        )

    return bags


def build_weak_conflict_audit(
    recording_ids,
    recording_to_indices,
    targets,
    weak_lookup,
    species,
):
    """Audit weak=0 cells that conflict with strong positives."""
    rows = []

    for recording_id in recording_ids:
        recording_id = str(recording_id)
        indices = np.asarray(
            recording_to_indices[recording_id],
            dtype=np.int64,
        )
        values = targets.iloc[indices][species].to_numpy(
            dtype=np.float32
        )
        strong_positive = (
            np.isfinite(values) & (values == 1)
        ).any(axis=0)

        for j, sp in enumerate(species):
            weak_value = weak_lookup.get((recording_id, sp))
            if weak_value is None:
                continue
            if float(weak_value) == 0.0 and bool(strong_positive[j]):
                rows.append({
                    "recording_id": recording_id,
                    "species": sp,
                    "weak_presence": 0,
                    "strong_positive_segments": int(
                        (np.isfinite(values[:, j]) & (values[:, j] == 1)).sum()
                    ),
                    "action": "mask_weak_negative",
                })

    return pd.DataFrame(rows)


def grouped_recording_split(
    recording_ids,
    seed,
    validation_fraction=0.20,
):
    rng = np.random.default_rng(seed)

    ids = np.asarray(
        list(recording_ids),
        dtype=object,
    )

    rng.shuffle(ids)

    n_val = max(
        1,
        int(
            round(
                len(ids)
                * validation_fraction
            )
        ),
    )

    val_ids = ids[:n_val]
    train_ids = ids[n_val:]

    return (
        train_ids.tolist(),
        val_ids.tolist(),
    )


def fit_scaler(
    X,
    segment_indices,
):
    """
    Fit standardization only on the training segments.

    This avoids using held-out site statistics.
    """
    if len(segment_indices) == 0:
        raise RuntimeError(
            "Cannot fit scaler with zero training segments."
        )

    arr = np.asarray(
        X[segment_indices],
        dtype=np.float32,
    )

    mean = arr.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = arr.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std[std < 1e-6] = 1.0

    return mean, std


def make_batch(
    bags,
    X,
    mean,
    std,
    n_species,
):
    max_len = max(
        len(b["indices"])
        for b in bags
    )

    xb = np.zeros(
        (
            len(bags),
            max_len,
            X.shape[1],
        ),
        dtype=np.float32,
    )

    mask = np.zeros(
        (
            len(bags),
            max_len,
        ),
        dtype=bool,
    )

    weak_y = np.full(
        (
            len(bags),
            n_species,
        ),
        np.nan,
        dtype=np.float32,
    )

    for b, bag in enumerate(bags):
        idx = bag["indices"]

        x = np.asarray(
            X[idx],
            dtype=np.float32,
        )

        x = (
            x - mean
        ) / std

        xb[
            b,
            :len(idx),
        ] = x

        mask[
            b,
            :len(idx),
        ] = True

        weak_y[b] = bag["weak_y"]

    return (
        xb,
        mask,
        weak_y,
    )


def batch_iter(
    bags,
    batch_size,
    rng,
    shuffle=True,
):
    order = np.arange(
        len(bags)
    )

    if shuffle:
        rng.shuffle(order)

    for start in range(
        0,
        len(order),
        batch_size,
    ):
        batch_indices = order[
            start:start + batch_size
        ]

        yield [
            bags[int(i)]
            for i in batch_indices
        ]


class AttentionMIL(tf.keras.Model):
    """
    Species-specific attention MIL.

    Input:
        x    [B,T,1024]
        mask [B,T]

    Outputs:
        segment_logits  [B,T,S]
        recording_logits[B,S]
        attention       [B,T,S]

    Recording logits are derived from the segment logits through
    attention-weighted log-mean-exp pooling.

    This keeps weak recording supervision connected to the segment classifier.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        attention_dim,
        n_species,
        dropout,
    ):
        super().__init__()

        self.encoder_dense1 = tf.keras.layers.Dense(
            hidden_dim,
            activation=None,
            name="encoder_dense1",
        )

        self.encoder_norm = tf.keras.layers.LayerNormalization(
            name="encoder_layernorm",
        )

        self.encoder_dense2 = tf.keras.layers.Dense(
            hidden_dim,
            activation=None,
            name="encoder_dense2",
        )

        self.dropout = tf.keras.layers.Dropout(
            dropout
        )

        self.segment_head = tf.keras.layers.Dense(
            n_species,
            name="segment_species_logits",
        )

        self.attention_hidden = tf.keras.layers.Dense(
            attention_dim,
            activation="tanh",
            name="attention_hidden",
        )

        self.attention_head = tf.keras.layers.Dense(
            n_species,
            use_bias=False,
            name="species_attention_logits",
        )

    def encode(
        self,
        x,
        training=False,
    ):
        h = self.encoder_dense1(x)
        h = self.encoder_norm(h)
        h = tf.nn.gelu(h)
        h = self.dropout(
            h,
            training=training,
        )

        h = self.encoder_dense2(h)
        h = tf.nn.gelu(h)
        h = self.dropout(
            h,
            training=training,
        )

        return h

    def call(
        self,
        inputs,
        training=False,
    ):
        x, mask = inputs

        h = self.encode(
            x,
            training=training,
        )

        segment_logits = self.segment_head(h)

        attention_logits = self.attention_head(
            self.attention_hidden(h)
        )

        # [B,T] -> [B,T,1]
        mask3 = tf.cast(
            mask[:, :, None],
            tf.float32,
        )

        masked_attention_logits = tf.where(
            mask[:, :, None],
            attention_logits,
            tf.fill(
                tf.shape(attention_logits),
                tf.constant(-1e9),
            ),
        )

        attention = tf.nn.softmax(
            masked_attention_logits,
            axis=1,
        )

        # The following is a normalized weighted log-mean-exp:
        #
        # log(sum_i softmax(a_i) * exp(z_i))
        #
        # This remains directly tied to segment logits and is more appropriate
        # for sparse events than simply averaging segment probabilities.
        weighted_exp = (
            attention
            * tf.exp(
                tf.clip_by_value(
                    segment_logits,
                    -20.0,
                    20.0,
                )
            )
        )

        recording_prob = tf.reduce_sum(
            weighted_exp,
            axis=1,
        )

        recording_prob = tf.clip_by_value(
            recording_prob,
            1e-6,
            1.0 - 1e-6,
        )

        recording_logits = tf.math.log(
            recording_prob
            / (
                1.0
                - recording_prob
            )
        )

        # Make sure padding cannot influence outputs.
        # Attention is already masked; this multiplication is only useful for
        # downstream diagnostics.
        attention = (
            attention
            * mask3
        )

        return (
            segment_logits,
            recording_logits,
            attention,
        )


def masked_bce(
    logits,
    target,
):
    observed = tf.math.is_finite(
        target
    )

    target_clean = tf.where(
        observed,
        target,
        tf.zeros_like(target),
    )

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_clean,
        logits=logits,
    )

    loss = tf.where(
        observed,
        loss,
        tf.zeros_like(loss),
    )

    denominator = tf.reduce_sum(
        tf.cast(
            observed,
            tf.float32,
        )
    )

    return tf.math.divide_no_nan(
        tf.reduce_sum(loss),
        denominator,
    )


def strong_positive_segment_loss(
    segment_logits,
    batch,
    targets,
    species,
):
    """
    Positive-only strong supervision.

    Only target == 1 contributes.

    NaN target cells contribute zero loss.
    """
    losses = []

    for b, bag in enumerate(batch):
        idx = bag["indices"]

        y = targets.iloc[
            idx
        ][species].to_numpy(
            dtype=np.float32
        )

        pos = (
            np.isfinite(y)
            & (y == 1)
        )

        if not pos.any():
            continue

        pos_t = tf.convert_to_tensor(
            pos.astype(bool)
        )

        logits = tf.boolean_mask(
            segment_logits[b],
            pos_t,
        )

        if tf.size(logits) == 0:
            continue

        losses.append(
            tf.reduce_mean(
                tf.nn.softplus(
                    -logits
                )
            )
        )

    if not losses:
        return tf.reduce_sum(
            segment_logits
        ) * 0.0

    return tf.reduce_mean(
        tf.stack(losses)
    )


def strong_recording_consistency_loss(
    segment_logits,
    recording_logits,
    batch,
    targets,
    species,
):
    """
    Require a recording-level prediction to explain confirmed strong
    segment evidence.

    This is one-sided:
        recording probability < strongest confirmed segment probability
        -> penalty

    It does NOT treat unlabeled segments as negatives.
    """
    seg_prob = tf.sigmoid(
        segment_logits
    )

    rec_prob = tf.sigmoid(
        recording_logits
    )

    penalties = []

    for b, bag in enumerate(batch):
        idx = bag["indices"]

        y = targets.iloc[
            idx
        ][species].to_numpy(
            dtype=np.float32
        )

        pos = (
            np.isfinite(y)
            & (y == 1)
        )

        if not pos.any():
            continue

        pos_t = tf.convert_to_tensor(
            pos.astype(bool)
        )

        positive_probs = tf.boolean_mask(
            seg_prob[b],
            pos_t,
        )

        strongest = tf.reduce_max(
            positive_probs,
            axis=0,
        )

        penalties.append(
            tf.reduce_mean(
                tf.nn.relu(
                    strongest
                    - rec_prob[b]
                )
            )
        )

    if not penalties:
        return tf.reduce_sum(
            recording_logits
        ) * 0.0

    return tf.reduce_mean(
        tf.stack(penalties)
    )


def make_optimizer(args):
    return tf.keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )


def forward_loss(
    model,
    batch,
    X,
    mean,
    std,
    targets,
    species,
    args,
    training,
):
    xb, mask, weak_y = make_batch(
        batch,
        X,
        mean,
        std,
        len(species),
    )

    xb = tf.convert_to_tensor(
        xb,
        dtype=tf.float32,
    )

    mask = tf.convert_to_tensor(
        mask,
        dtype=tf.bool,
    )

    weak_y = tf.convert_to_tensor(
        weak_y,
        dtype=tf.float32,
    )

    (
        segment_logits,
        recording_logits,
        attention,
    ) = model(
        (xb, mask),
        training=training,
    )

    weak_loss = masked_bce(
        recording_logits,
        weak_y,
    )

    strong_loss = (
        strong_positive_segment_loss(
            segment_logits,
            batch,
            targets,
            species,
        )
    )

    consistency_loss = (
        strong_recording_consistency_loss(
            segment_logits,
            recording_logits,
            batch,
            targets,
            species,
        )
    )

    total = (
        args.weak_loss_weight
        * weak_loss
        + args.strong_loss_weight
        * strong_loss
        + args.consistency_weight
        * consistency_loss
    )

    return (
        total,
        weak_loss,
        strong_loss,
        consistency_loss,
        segment_logits,
        recording_logits,
        attention,
    )


@tf.function(
    reduce_retracing=True
)
def train_step(
    model,
    optimizer,
    xb,
    mask,
    weak_y,
    strong_mask,
    strong_weights,
    weak_weight,
    strong_weight,
    consistency_weight,
):
    """
    TensorFlow graph-mode training step.

    The strong_mask has shape [B,T,S] and contains only confirmed
    strong-positive cells.

    This implementation avoids converting NaN labels into negatives.
    """
    with tf.GradientTape() as tape:
        (
            segment_logits,
            recording_logits,
            attention,
        ) = model(
            (xb, mask),
            training=True,
        )

        observed = tf.math.is_finite(
            weak_y
        )

        weak_clean = tf.where(
            observed,
            weak_y,
            tf.zeros_like(weak_y),
        )

        weak_raw = (
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=weak_clean,
                logits=recording_logits,
            )
        )

        weak_raw = tf.where(
            observed,
            weak_raw,
            tf.zeros_like(weak_raw),
        )

        weak_denominator = tf.reduce_sum(
            tf.cast(
                observed,
                tf.float32,
            )
        )

        weak_loss = tf.math.divide_no_nan(
            tf.reduce_sum(
                weak_raw
            ),
            weak_denominator,
        )

        # Positive-only strong segment loss.
        strong_raw = tf.nn.softplus(
            -segment_logits
        )

        # strong_mask is boolean; cast it before multiplying with
        # floating-point tensors. TensorFlow 2.21 does not implicitly
        # multiply bool and float tensors.
        strong_mask_float = tf.cast(
            strong_mask,
            tf.float32,
        )

        strong_raw = (
            strong_raw
            * strong_mask_float
            * strong_weights
        )

        strong_denominator = tf.reduce_sum(
            strong_mask_float
            * strong_weights
        )

        strong_loss = tf.math.divide_no_nan(
            tf.reduce_sum(
                strong_raw
            ),
            strong_denominator,
        )

        seg_prob = tf.sigmoid(
            segment_logits
        )

        rec_prob = tf.sigmoid(
            recording_logits
        )

        # For each species, the strongest confirmed positive segment should
        # not be more positive than the recording-level prediction without
        # paying a consistency penalty.
        masked_positive_prob = tf.where(
            strong_mask,
            seg_prob,
            tf.zeros_like(seg_prob),
        )

        strongest = tf.reduce_max(
            masked_positive_prob,
            axis=1,
        )

        has_positive = tf.reduce_any(
            strong_mask,
            axis=1,
        )

        consistency_raw = tf.nn.relu(
            strongest
            - rec_prob
        )

        consistency_raw = tf.where(
            has_positive,
            consistency_raw,
            tf.zeros_like(consistency_raw),
        )

        consistency_denominator = tf.reduce_sum(
            tf.cast(
                has_positive,
                tf.float32,
            )
        )

        consistency_loss = tf.math.divide_no_nan(
            tf.reduce_sum(
                consistency_raw
            ),
            consistency_denominator,
        )

        total_loss = (
            weak_weight
            * weak_loss
            + strong_weight
            * strong_loss
            + consistency_weight
            * consistency_loss
        )

    variables = model.trainable_variables

    gradients = tape.gradient(
        total_loss,
        variables,
    )

    gradients = [
        tf.clip_by_norm(
            g,
            5.0,
        )
        if g is not None
        else None
        for g in gradients
    ]

    optimizer.apply_gradients(
        zip(
            gradients,
            variables,
        )
    )

    return (
        total_loss,
        weak_loss,
        strong_loss,
        consistency_loss,
    )


def prepare_strong_mask(
    batch,
    targets,
    species,
):
    """
    Build [B,T,S] boolean mask:
        True only where source_type == strong_event AND target == 1.

    This explicitly prevents weak_candidate segments from receiving
    strong-label loss.
    """
    max_len = max(
        len(b["indices"])
        for b in batch
    )

    mask = np.zeros(
        (
            len(batch),
            max_len,
            len(species),
        ),
        dtype=bool,
    )

    for b, bag in enumerate(batch):
        idx = bag["indices"]

        source = (
            targets.iloc[idx]
            ["source_type"]
            .astype(str)
            .eq("strong_event")
            .to_numpy()
        )

        y = targets.iloc[
            idx
        ][species].to_numpy(
            dtype=np.float32
        )

        positive = (
            np.isfinite(y)
            & (y == 1)
        )

        mask[
            b,
            :len(idx),
        ] = (
            positive
            & source[:, None]
        )

    return mask


def compute_validation_predictions(
    model,
    bags,
    X,
    mean,
    std,
    species,
    batch_size,
):
    scores = []

    rng = np.random.default_rng(
        SEED
    )

    for batch in batch_iter(
        bags,
        batch_size,
        rng,
        shuffle=False,
    ):
        xb, mask, _ = make_batch(
            batch,
            X,
            mean,
            std,
            len(species),
        )

        (
            _,
            recording_logits,
            _,
        ) = model(
            (
                tf.convert_to_tensor(
                    xb,
                    dtype=tf.float32,
                ),
                tf.convert_to_tensor(
                    mask,
                    dtype=tf.bool,
                ),
            ),
            training=False,
        )

        scores.append(
            tf.sigmoid(
                recording_logits
            ).numpy()
        )

    if not scores:
        return np.empty(
            (
                0,
                len(species),
            ),
            dtype=np.float32,
        )

    return np.vstack(
        scores
    )


def predict_test(
    model,
    bags,
    X,
    mean,
    std,
    species,
    batch_size,
):
    recording_scores = []
    recording_ids = []
    attention_outputs = []

    rng = np.random.default_rng(
        SEED
    )

    for batch in batch_iter(
        bags,
        batch_size,
        rng,
        shuffle=False,
    ):
        xb, mask, _ = make_batch(
            batch,
            X,
            mean,
            std,
            len(species),
        )

        (
            segment_logits,
            recording_logits,
            attention,
        ) = model(
            (
                tf.convert_to_tensor(
                    xb,
                    dtype=tf.float32,
                ),
                tf.convert_to_tensor(
                    mask,
                    dtype=tf.bool,
                ),
            ),
            training=False,
        )

        recording_scores.append(
            tf.sigmoid(
                recording_logits
            ).numpy()
        )

        recording_ids.extend(
            [
                b["recording_id"]
                for b in batch
            ]
        )

        attention_outputs.extend(
            attention.numpy()
        )

    return (
        np.vstack(
            recording_scores
        ),
        recording_ids,
        attention_outputs,
    )


def choose_threshold(
    y,
    scores,
):
    valid = np.isfinite(y)

    if valid.sum() == 0:
        return 0.5

    yv = y[
        valid
    ].astype(int)

    sv = scores[
        valid
    ]

    if yv.sum() == 0:
        return 0.5

    candidates = np.unique(
        np.concatenate(
            [
                np.arange(
                    0.05,
                    1.0,
                    0.05,
                ),
                sv,
            ]
        )
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidates:
        pred = (
            sv >= threshold
        ).astype(int)

        value = f1_score(
            yv,
            pred,
            zero_division=0,
        )

        if value > best_f1:
            best_f1 = value
            best_threshold = float(
                threshold
            )

    return best_threshold


def safe_ap(y, score):
    try:
        return float(
            average_precision_score(
                y,
                score,
            )
        )
    except ValueError:
        return np.nan


def safe_auc(y, score):
    try:
        return float(
            roc_auc_score(
                y,
                score,
            )
        )
    except ValueError:
        return np.nan


def evaluate_recording_predictions(
    site,
    species,
    bags,
    scores,
    thresholds,
    support,
):
    y = np.vstack(
        [
            b["weak_y"]
            for b in bags
        ]
    )

    rows = []

    for j, sp in enumerate(species):
        valid = np.isfinite(
            y[:, j]
        )

        if not valid.any():
            continue

        yy = y[
            valid,
            j,
        ].astype(int)

        ss = scores[
            valid,
            j,
        ]

        pred = (
            ss >= thresholds[j]
        ).astype(int)

        train_positive = int(
            support.loc[
                support.species.eq(sp),
                "train_positive_segments",
            ].iloc[0]
        )

        test_positive_recordings = int(yy.sum())

        if train_positive == 0:
            status = "spatially_novel"
        elif train_positive < MIN_STANDARD_SUPPORT:
            status = "low_support"
        else:
            status = "standard_support"

        # A species with zero held-out positives cannot contribute a meaningful
        # positive-class F1/AP/AUC estimate. Keep the row for transparency,
        # but exclude it from the primary macro-metric denominator.
        if test_positive_recordings > 0:
            test_precision = precision_score(
                yy,
                pred,
                zero_division=0,
            )
            test_recall = recall_score(
                yy,
                pred,
                zero_division=0,
            )
            test_f1 = f1_score(
                yy,
                pred,
                zero_division=0,
            )
            test_ap = safe_ap(yy, ss)
            test_auc = safe_auc(yy, ss)
        else:
            status = "no_test_positives"
            test_precision = np.nan
            test_recall = np.nan
            test_f1 = np.nan
            test_ap = np.nan
            test_auc = np.nan

        rows.append(
            {
                "held_out_site": site,
                "species": sp,
                "status": status,
                "train_positive_segments": train_positive,
                "test_recordings_with_weak_label": int(valid.sum()),
                "test_positive_recordings": test_positive_recordings,
                "threshold": float(thresholds[j]),
                "precision": test_precision,
                "recall": test_recall,
                "f1": test_f1,
                "average_precision": test_ap,
                "roc_auc": test_auc,
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_segment_proxy(
    site,
    metadata,
    targets,
    species,
    strong_test_indices,
    model,
    bags,
    X,
    mean,
    std,
    batch_size,
    support,
):
    """
    Secondary diagnostic only.

    It evaluates strong-event test segments by treating:
        target == 1 -> positive
        target != 1 -> proxy negative

    Because the latter includes unknown labels, this MUST NOT be interpreted
    as a fully supervised segment-level test metric.
    """
    # Map global segment index -> predicted segment probability.
    pred_by_index = {}

    rng = np.random.default_rng(
        SEED
    )

    for batch in batch_iter(
        bags,
        batch_size,
        rng,
        shuffle=False,
    ):
        xb, mask, _ = make_batch(
            batch,
            X,
            mean,
            std,
            len(species),
        )

        (
            segment_logits,
            _,
            _,
        ) = model(
            (
                tf.convert_to_tensor(
                    xb,
                    dtype=tf.float32,
                ),
                tf.convert_to_tensor(
                    mask,
                    dtype=tf.bool,
                ),
            ),
            training=False,
        )

        probabilities = tf.sigmoid(
            segment_logits
        ).numpy()

        for b, bag in enumerate(
            batch
        ):
            idx = bag["indices"]

            for t, global_idx in enumerate(
                idx
            ):
                pred_by_index[
                    int(global_idx)
                ] = probabilities[
                    b,
                    t,
                ]

    rows = []

    strong_test_indices = np.asarray(
        strong_test_indices,
        dtype=int,
    )

    for j, sp in enumerate(species):
        y_original = targets.iloc[
            strong_test_indices
        ][sp].to_numpy(
            dtype=np.float32
        )

        # Diagnostic proxy: confirmed positive vs all other strong-event rows.
        y = (
            np.isfinite(
                y_original
            )
            & (y_original == 1)
        ).astype(int)

        scores = np.asarray(
            [
                pred_by_index[
                    int(idx)
                ][j]
                for idx in strong_test_indices
            ],
            dtype=float,
        )

        train_positive = int(
            support.loc[
                support.species.eq(sp),
                "train_positive_segments",
            ].iloc[0]
        )

        test_positive = int(
            y.sum()
        )

        if test_positive == 0:
            continue

        if train_positive == 0:
            status = "spatially_novel"
        elif train_positive < MIN_STANDARD_SUPPORT:
            status = "low_support"
        else:
            status = "standard_support"

        rows.append(
            {
                "held_out_site": site,
                "species": sp,
                "status": status,
                "train_positive_segments": train_positive,
                "test_positive_segments": test_positive,
                "proxy_average_precision": safe_ap(
                    y,
                    scores,
                ),
                "proxy_roc_auc": safe_auc(
                    y,
                    scores,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def species_support(
    targets,
    indices,
    species,
):
    rows = []

    for sp in species:
        positive = (
            targets.iloc[
                indices
            ][sp]
            .notna()
            & targets.iloc[
                indices
            ][sp].eq(1)
        )

        rows.append(
            {
                "species": sp,
                "train_positive_segments": int(
                    positive.sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_attention_topk(
    output_path,
    held_out_site,
    test_bags,
    attention,
    metadata,
    species,
    top_k=3,
):
    rows = []

    for b, bag in enumerate(
        test_bags
    ):
        idx = bag["indices"]
        att = attention[b][
            :len(idx)
        ]

        for j, sp in enumerate(
            species
        ):
            order = np.argsort(
                -att[:, j]
            )[:top_k]

            for rank, local_idx in enumerate(
                order,
                start=1,
            ):
                global_idx = int(
                    idx[local_idx]
                )

                rows.append(
                    {
                        "held_out_site": held_out_site,
                        "recording_id": bag[
                            "recording_id"
                        ],
                        "rank": rank,
                        "species": sp,
                        "segment_id": str(
                            metadata.iloc[
                                global_idx
                            ]["segment_id"]
                        ),
                        "attention_weight": float(
                            att[
                                local_idx,
                                j,
                            ]
                        ),
                    }
                )

    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False,
    )


def train_fold(
    model,
    train_bags,
    val_bags,
    X,
    mean,
    std,
    targets,
    species,
    args,
    fold_seed,
):
    optimizer = make_optimizer(
        args
    )

    # Keras 3 / TensorFlow 2.21 creates Adam's slot variables lazily on the
    # first apply_gradients() call. Because train_step is a tf.function and
    # this script trains a fresh model/optimizer for every LOSO fold, leaving
    # optimizer slots to be created inside train_step causes the second fold
    # to fail with:
    #   ValueError: tf.function only supports singleton tf.Variables created
    #   on the first call.
    #
    # Build the optimizer state explicitly for THIS fold's model variables
    # before train_step is traced/called.
    optimizer.build(model.trainable_variables)

    rng = np.random.default_rng(
        fold_seed
    )

    val_y = np.vstack(
        [
            b["weak_y"]
            for b in val_bags
        ]
    )

    best_val_ap = -np.inf
    best_weights = None
    patience = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        losses = []
        weak_losses = []
        strong_losses = []
        consistency_losses = []

        for batch in batch_iter(
            train_bags,
            args.batch_size,
            rng,
            shuffle=True,
        ):
            xb, mask, weak_y = make_batch(
                batch,
                X,
                mean,
                std,
                len(species),
            )

            strong_mask = prepare_strong_mask(
                batch,
                targets,
                species,
            )

            # The strong loss is only applied to positive strong cells.
            # Equal weighting here is deliberate: the loss is positive-only.
            strong_weights = np.ones(
                strong_mask.shape,
                dtype=np.float32,
            )

            result = train_step(
                model,
                optimizer,
                tf.convert_to_tensor(
                    xb,
                    dtype=tf.float32,
                ),
                tf.convert_to_tensor(
                    mask,
                    dtype=tf.bool,
                ),
                tf.convert_to_tensor(
                    weak_y,
                    dtype=tf.float32,
                ),
                tf.convert_to_tensor(
                    strong_mask,
                    dtype=tf.bool,
                ),
                tf.convert_to_tensor(
                    strong_weights,
                    dtype=tf.float32,
                ),
                tf.constant(
                    args.weak_loss_weight,
                    dtype=tf.float32,
                ),
                tf.constant(
                    args.strong_loss_weight,
                    dtype=tf.float32,
                ),
                tf.constant(
                    args.consistency_weight,
                    dtype=tf.float32,
                ),
            )

            total, weak_loss, strong_loss, consistency = result

            losses.append(
                float(total.numpy())
            )
            weak_losses.append(
                float(weak_loss.numpy())
            )
            strong_losses.append(
                float(strong_loss.numpy())
            )
            consistency_losses.append(
                float(consistency.numpy())
            )

        val_scores = compute_validation_predictions(
            model,
            val_bags,
            X,
            mean,
            std,
            species,
            args.batch_size,
        )

        aps = []

        for j in range(
            len(species)
        ):
            valid = np.isfinite(
                val_y[:, j]
            )

            if (
                valid.sum() > 0
                and val_y[
                    valid,
                    j,
                ].sum() > 0
            ):
                aps.append(
                    safe_ap(
                        val_y[
                            valid,
                            j,
                        ].astype(int),
                        val_scores[
                            valid,
                            j,
                        ],
                    )
                )

        val_ap = (
            float(
                np.nanmean(aps)
            )
            if aps
            else -np.inf
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap

            best_weights = (
                model.get_weights()
            )

            patience = 0
        else:
            patience += 1

        if (
            epoch == 1
            or epoch % 5 == 0
        ):
            print(
                f"    epoch {epoch:03d} | "
                f"loss={np.mean(losses):.4f} | "
                f"weak={np.mean(weak_losses):.4f} | "
                f"strong={np.mean(strong_losses):.4f} | "
                f"consistency={np.mean(consistency_losses):.4f} | "
                f"val weak macro-AP={val_ap:.4f}"
            )

        if (
            patience
            >= args.early_stopping_patience
        ):
            print(
                f"    early stopping at epoch {epoch}"
            )
            break

    if best_weights is not None:
        model.set_weights(
            best_weights
        )

    val_scores = compute_validation_predictions(
        model,
        val_bags,
        X,
        mean,
        std,
        species,
        args.batch_size,
    )

    thresholds = np.full(
        len(species),
        0.5,
        dtype=float,
    )

    validation_diagnostics = []

    for j, sp in enumerate(species):
        valid = np.isfinite(val_y[:, j])

        if valid.sum() == 0:
            validation_positive_recordings = 0
            validation_f1 = np.nan
        else:
            yv = val_y[valid, j].astype(int)
            sv = val_scores[valid, j]
            validation_positive_recordings = int(yv.sum())

            thresholds[j] = choose_threshold(
                val_y[:, j],
                val_scores[:, j],
            )

            if validation_positive_recordings > 0:
                validation_f1 = f1_score(
                    yv,
                    (sv >= thresholds[j]).astype(int),
                    zero_division=0,
                )
            else:
                validation_f1 = np.nan

        validation_diagnostics.append(
            {
                "species": sp,
                "validation_recordings_with_weak_label": int(valid.sum()),
                "validation_positive_recordings": validation_positive_recordings,
                "selected_threshold": float(thresholds[j]),
                "validation_f1": validation_f1,
            }
        )

    validation_diagnostics = pd.DataFrame(validation_diagnostics)

    return (
        model,
        thresholds,
        best_val_ap,
        validation_diagnostics,
    )


def save_model(
    model,
    path,
):
    """
    Save Keras weights.

    The model architecture is deterministic from run_config.json.
    """
    model.save_weights(
        str(path)
    )


def main():
    args = parse_args()

    set_seed(
        SEED
    )

    configure_tensorflow(
        args
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        X,
        embedding_metadata,
        metadata,
        targets,
        species,
        weak,
    ) = load_data(
        args
    )

    sites = sorted(
        metadata[
            "site_id"
        ].astype(str).unique()
    )

    recording_to_indices = (
        build_recording_index(
            metadata
        )
    )

    weak_lookup = {
        (
            str(row.recording_id),
            str(row.species),
        ): float(row.presence)
        for row in weak.itertuples(
            index=False
        )
    }

    # Audit weak-negative / strong-positive conflicts. These cells are
    # masked from weak supervision during bag construction, while the
    # confirmed strong positives remain supervised by the strong loss.
    conflict_audit = build_weak_conflict_audit(
        metadata["recording_id"].astype(str).unique().tolist(),
        recording_to_indices,
        targets,
        weak_lookup,
        species,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    conflict_audit.to_csv(
        args.output / "weak_strong_conflicts_masked.csv",
        index=False,
    )

    print(
        f"Weak-negative/strong-positive conflicts masked: {len(conflict_audit):,}"
    )

    run_config = {
        "method": "TensorFlow_attention_MIL_with_strong_positive_and_weak_recording_supervision",
        "seed": SEED,
        "tensorflow_version": tf.__version__,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "attention_dim": args.attention_dim,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "weak_loss_weight": args.weak_loss_weight,
        "strong_loss_weight": args.strong_loss_weight,
        "consistency_weight": args.consistency_weight,
        "early_stopping_patience": args.early_stopping_patience,
        "max_segments_per_recording": args.max_segments_per_recording,
        "n_segments": len(metadata),
        "n_species": len(species),
        "sites": sites,
        "species": species,
        "embeddings": str(args.embeddings),
        "embedding_metadata": str(args.embedding_metadata),
        "metadata": str(args.metadata),
        "targets": str(args.targets),
        "weak_long": str(args.weak_long),
        "weak_clean": str(args.weak_clean),
        "output": str(args.output),
    }

    (
        args.output
        / "run_config.json"
    ).write_text(
        json.dumps(
            run_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    all_recording_results = []
    all_segment_results = []
    fold_summaries = []
    novelty_rows = []

    for fold_no, held_out_site in enumerate(
        sites
    ):
        print()
        print("=" * 80)
        print(
            f"HELD-OUT SITE: {held_out_site}"
        )
        print("=" * 80)

        held_out_mask = (
            metadata[
                "site_id"
            ].astype(str)
            .eq(held_out_site)
            .to_numpy()
        )

        training_mask = (
            ~held_out_mask
        )

        training_recordings = sorted(
            metadata.loc[
                training_mask,
                "recording_id",
            ]
            .astype(str)
            .unique()
        )

        test_recordings = sorted(
            metadata.loc[
                held_out_mask,
                "recording_id",
            ]
            .astype(str)
            .unique()
        )

        fit_recordings, val_recordings = (
            grouped_recording_split(
                training_recordings,
                seed=SEED + fold_no,
            )
        )

        # ---------------------------------------------------------------
        # CRITICAL MIL DATA FLOW
        # ---------------------------------------------------------------
        # Bags contain ALL segments from the selected recordings.
        # We do NOT restrict these indices to strong_event.
        # ---------------------------------------------------------------
        train_bags = build_bags(
            fit_recordings,
            recording_to_indices,
            metadata,
            targets,
            weak_lookup,
            species,
            args.max_segments_per_recording,
            SEED + 1000 + fold_no,
        )

        val_bags = build_bags(
            val_recordings,
            recording_to_indices,
            metadata,
            targets,
            weak_lookup,
            species,
            args.max_segments_per_recording,
            SEED + 2000 + fold_no,
        )

        test_bags = build_bags(
            test_recordings,
            recording_to_indices,
            metadata,
            targets,
            weak_lookup,
            species,
            args.max_segments_per_recording,
            SEED + 3000 + fold_no,
        )

        # Segment indices are kept separately for strong-support accounting
        # and the secondary strong-event test diagnostic.
        fit_segment_indices = np.concatenate(
            [
                bag["indices"]
                for bag in train_bags
            ]
        )

        strong_fit_indices = fit_segment_indices[
            metadata.iloc[
                fit_segment_indices
            ]["source_type"]
            .astype(str)
            .eq("strong_event")
            .to_numpy()
        ]

        strong_test_indices = np.concatenate(
            [
                bag["indices"]
                for bag in test_bags
            ]
        )

        strong_test_indices = strong_test_indices[
            metadata.iloc[
                strong_test_indices
            ]["source_type"]
            .astype(str)
            .eq("strong_event")
            .to_numpy()
        ]

        support = species_support(
            targets,
            strong_fit_indices,
            species,
        )

        test_support = species_support(
            targets,
            strong_test_indices,
            species,
        ).rename(
            columns={
                "train_positive_segments":
                    "test_positive_segments"
            }
        )

        status = support.merge(
            test_support,
            on="species",
            how="left",
        )

        status["held_out_site"] = (
            held_out_site
        )

        status["status"] = "standard_support"

        status.loc[
            status[
                "test_positive_segments"
            ].eq(0),
            "status",
        ] = "no_test_positives"

        status.loc[
            (
                status[
                    "test_positive_segments"
                ].gt(0)
                & status[
                    "train_positive_segments"
                ].eq(0)
            ),
            "status",
        ] = "spatially_novel"

        status.loc[
            (
                status[
                    "test_positive_segments"
                ].gt(0)
                & status[
                    "train_positive_segments"
                ].gt(0)
                & status[
                    "train_positive_segments"
                ].lt(
                    MIN_STANDARD_SUPPORT
                )
            ),
            "status",
        ] = "low_support"

        status = status[
            [
                "held_out_site",
                "species",
                "train_positive_segments",
                "test_positive_segments",
                "status",
            ]
        ]

        status.to_csv(
            args.output
            / f"species_status_{held_out_site}.csv",
            index=False,
        )

        print()
        print(
            "Species support status:"
        )
        print(
            status[
                "status"
            ].value_counts()
            .to_string()
        )

        # Report actual bag source composition.
        train_all = np.concatenate(
            [
                b["indices"]
                for b in train_bags
            ]
        )

        print()
        print(
            "Training MIL bag source distribution:"
        )
        print(
            metadata.iloc[
                train_all
            ]["source_type"]
            .value_counts()
            .to_string()
        )

        # ---------------------------------------------------------------
        # Fit scaler using TRAINING BAG SEGMENTS ONLY.
        # ---------------------------------------------------------------
        mean, std = fit_scaler(
            X,
            fit_segment_indices,
        )

        np.save(
            args.output
            / f"scaler_mean_{held_out_site}.npy",
            mean,
        )

        np.save(
            args.output
            / f"scaler_std_{held_out_site}.npy",
            std,
        )

        # ---------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------
        model = AttentionMIL(
            input_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            n_species=len(species),
            dropout=args.dropout,
        )

        # Build model variables before save/load.
        dummy_x = tf.zeros(
            (
                1,
                1,
                X.shape[1],
            ),
            dtype=tf.float32,
        )

        dummy_mask = tf.ones(
            (
                1,
                1,
            ),
            dtype=tf.bool,
        )

        model(
            (
                dummy_x,
                dummy_mask,
            ),
            training=False,
        )

        (
            model,
            thresholds,
            best_val_ap,
            validation_diagnostics,
        ) = train_fold(
            model,
            train_bags,
            val_bags,
            X,
            mean,
            std,
            targets,
            species,
            args,
            SEED + fold_no * 100,
        )

        save_model(
            model,
            args.output
            / f"model_{held_out_site}.weights.h5",
        )

        threshold_json = {
            sp: float(
                thresholds[j]
            )
            for j, sp in enumerate(
                species
            )
        }

        # Validation-only threshold diagnostics. These values are produced
        # entirely from the training-site validation recordings.
        validation_diagnostics = validation_diagnostics.copy()
        validation_diagnostics["held_out_site"] = held_out_site
        validation_diagnostics["validation_macro_average_precision"] = best_val_ap

        validation_diagnostics.to_csv(
            args.output / f"threshold_diagnostics_{held_out_site}.csv",
            index=False,
        )

        (
            args.output
            / f"thresholds_{held_out_site}.json"
        ).write_text(
            json.dumps(
                threshold_json,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ---------------------------------------------------------------
        # Test predictions
        # ---------------------------------------------------------------
        test_scores, test_ids, attention = (
            predict_test(
                model,
                test_bags,
                X,
                mean,
                std,
                species,
                args.batch_size,
            )
        )

        recording_results = (
            evaluate_recording_predictions(
                held_out_site,
                species,
                test_bags,
                test_scores,
                thresholds,
                support,
            )
        )

        recording_results = recording_results.merge(
            validation_diagnostics[
                [
                    "species",
                    "validation_recordings_with_weak_label",
                    "validation_positive_recordings",
                    "selected_threshold",
                    "validation_f1",
                ]
            ],
            on="species",
            how="left",
        )
        recording_results[
            "validation_macro_average_precision"
        ] = best_val_ap

        all_recording_results.append(
            recording_results
        )

        # ---------------------------------------------------------------
        # Recording prediction table
        # ---------------------------------------------------------------
        prediction_df = pd.DataFrame(
            {
                "recording_id": test_ids,
                "held_out_site": held_out_site,
            }
        )

        for j, sp in enumerate(
            species
        ):
            prediction_df[
                f"{sp}__score"
            ] = test_scores[:, j]

            prediction_df[
                f"{sp}__pred"
            ] = (
                test_scores[:, j]
                >= thresholds[j]
            ).astype(int)

        prediction_df.to_csv(
            args.output
            / f"recording_predictions_{held_out_site}.csv",
            index=False,
        )

        # ---------------------------------------------------------------
        # Attention interpretability
        # ---------------------------------------------------------------
        save_attention_topk(
            args.output
            / f"attention_top3_{held_out_site}.csv",
            held_out_site,
            test_bags,
            attention,
            metadata,
            species,
            top_k=3,
        )

        # ---------------------------------------------------------------
        # Secondary strong-event segment proxy
        # ---------------------------------------------------------------
        segment_results = (
            evaluate_segment_proxy(
                held_out_site,
                metadata,
                targets,
                species,
                strong_test_indices,
                model,
                test_bags,
                X,
                mean,
                std,
                args.batch_size,
                support,
            )
        )

        if not segment_results.empty:
            all_segment_results.append(
                segment_results
            )

        conventional = recording_results[
            recording_results.status.isin(
                [
                    "standard_support",
                    "low_support",
                ]
            )
            & (recording_results["test_positive_recordings"] > 0)
        ]

        if not conventional.empty:
            fold_summaries.append(
                {
                    "held_out_site": held_out_site,
                    "n_species": len(
                        conventional
                    ),
                    "macro_f1": conventional[
                        "f1"
                    ].mean(),
                    "macro_precision": conventional[
                        "precision"
                    ].mean(),
                    "macro_recall": conventional[
                        "recall"
                    ].mean(),
                    "macro_average_precision": conventional[
                        "average_precision"
                    ].mean(),
                    "macro_roc_auc": conventional[
                        "roc_auc"
                    ].mean(),
                    "median_train_positive_segments": conventional[
                        "train_positive_segments"
                    ].median(),
                }
            )

        novel = status[
            status[
                "status"
            ].eq("spatially_novel")
        ]

        novelty_rows.append(
            {
                "held_out_site": held_out_site,
                "spatially_novel_species": len(
                    novel
                ),
                "novel_test_positive_segments": int(
                    novel[
                        "test_positive_segments"
                    ].sum()
                ),
            }
        )

        print()
        print(
            "Recording-level weak-label results:"
        )

        if recording_results.empty:
            print(
                "  No weak-label test metrics available."
            )
        else:
            print(
                recording_results[
                    [
                        "species",
                        "status",
                        "train_positive_segments",
                        "validation_positive_recordings",
                        "selected_threshold",
                        "validation_f1",
                        "test_positive_recordings",
                        "f1",
                        "average_precision",
                        "roc_auc",
                    ]
                ].to_string(
                    index=False
                )
            )

    # ---------------------------------------------------------------
    # Final outputs
    # ---------------------------------------------------------------
    if all_recording_results:
        recording_results_all = pd.concat(
            all_recording_results,
            ignore_index=True,
        )
    else:
        recording_results_all = pd.DataFrame()

    if all_segment_results:
        segment_results_all = pd.concat(
            all_segment_results,
            ignore_index=True,
        )
    else:
        segment_results_all = pd.DataFrame()

    summary = pd.DataFrame(
        fold_summaries
    )

    if not recording_results_all.empty:
        conventional_all = (
            recording_results_all[
                recording_results_all[
                    "status"
                ].isin(
                    [
                        "standard_support",
                        "low_support",
                    ]
                )
                & (
                    recording_results_all["test_positive_recordings"] > 0
                )
            ]
        )

        if not conventional_all.empty:
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "held_out_site": "ALL_FOLDS",
                                "n_species": len(
                                    conventional_all
                                ),
                                "macro_f1": conventional_all[
                                    "f1"
                                ].mean(),
                                "macro_precision": conventional_all[
                                    "precision"
                                ].mean(),
                                "macro_recall": conventional_all[
                                    "recall"
                                ].mean(),
                                "macro_average_precision": conventional_all[
                                    "average_precision"
                                ].mean(),
                                "macro_roc_auc": conventional_all[
                                    "roc_auc"
                                ].mean(),
                                "median_train_positive_segments": conventional_all[
                                    "train_positive_segments"
                                ].median(),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    recording_results_all.to_csv(
        args.output
        / "loso_recording_species_results.csv",
        index=False,
    )

    # Compact threshold-vs-test diagnostic requested for separating
    # threshold/calibration failures from ranking/representation failures.
    if not recording_results_all.empty:
        threshold_diagnostics_all = recording_results_all[
            [
                "held_out_site",
                "species",
                "status",
                "train_positive_segments",
                "validation_recordings_with_weak_label",
                "validation_positive_recordings",
                "selected_threshold",
                "validation_f1",
                "test_recordings_with_weak_label",
                "test_positive_recordings",
                "f1",
                "average_precision",
                "roc_auc",
            ]
        ].copy()

        threshold_diagnostics_all.to_csv(
            args.output / "threshold_diagnostics_all_folds.csv",
            index=False,
        )

    segment_results_all.to_csv(
        args.output
        / "loso_segment_proxy_results.csv",
        index=False,
    )

    summary.to_csv(
        args.output
        / "loso_recording_summary.csv",
        index=False,
    )

    # Make the evaluation denominator explicit. Species with zero positive
    # held-out recordings are not part of the primary macro-F1/AP/AUC
    # denominator, but they are retained here for transparent reporting.
    if not recording_results_all.empty:
        recording_results_all[
            recording_results_all["test_positive_recordings"] > 0
        ].to_csv(
            args.output / "species_evaluable_test_results.csv",
            index=False,
        )
        recording_results_all[
            recording_results_all["test_positive_recordings"] == 0
        ].to_csv(
            args.output / "species_no_test_positives.csv",
            index=False,
        )

    novelty = pd.DataFrame(
        novelty_rows
    )

    novelty.to_csv(
        args.output
        / "spatial_novelty_summary.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL LOSO RECORDING-LEVEL SUMMARY")
    print("=" * 80)

    if summary.empty:
        print(
            "No conventional recording-level metrics were available."
        )
    else:
        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print(
        "Spatially novel species:"
    )

    print(
        novelty.to_string(
            index=False
        )
    )

    print()
    print(
        "Secondary segment-level results are saved separately as:"
    )
    print(
        args.output
        / "loso_segment_proxy_results.csv"
    )

    print()
    print(
        "Results written to:"
    )
    print(
        args.output
    )


if __name__ == "__main__":
    main()
