#!/usr/bin/env python3
"""
Build a 3-second multi-label anuran acoustic dataset.

Inputs
------
1. Audio recordings, searched recursively under --audio-dir.
2. Strong annotation files under --label-dir, one per recording:
       START<TAB>END<TAB>LABEL
   Short annotations (<= --strong-max-duration) become strong event clips.
3. Optional recording-level weak labels CSV supplied with --weak-labels.
   The CSV must contain AUDIO_FILE_ID plus SPECIES_<species_code> columns.
   Values are preserved as ordinal calling-activity values (0=absence,
   1=Low, 2=Moderate, 3=High). They are NOT assigned as segment labels.

Outputs
-------
output/
  strong/*.wav
  candidates/*.wav
  metadata.csv
  recording_splits.csv
  recording_weak_labels.csv
  species_vocabulary.csv
  site_inventory.csv
  build_summary.txt

Key design choices
------------------
- Final biological targets are multi-label species, not recording-quality classes.
- Strong labels ending in _L/_M/_H are parsed as species + recording quality.
- Multiple strong species overlapping one 3-s window are merged into one
  multi-label target.
- Weak labels remain recording-level ordinal supervision. Candidate 3-s clips
  do not inherit those values as segment-level targets.
- Train/val/test assignment is by recording ID only. Site identity is retained
  in every recording and segment row, but spatial holdout is deliberately NOT
  performed during segment construction.
- This makes the builder independent of the number of sites. Leave-one-site-out
  or other spatial folds should be generated afterward from metadata.csv.
- The weak-label matrix is written separately so the original 0-3 values are
  retained exactly and are not duplicated into every segment as target values.
"""

# Version: site-agnostic builder; spatial folds are created post-build.

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
from datetime import datetime
import re
from collections import Counter

import soundfile as sf


QUALITY_CODES = {"L", "M", "H"}
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def parse_args():
    p = argparse.ArgumentParser(description="Build a 3-second multi-label anuran dataset.")
    p.add_argument("--audio-dir", type=Path, default=Path("audio"))
    p.add_argument("--label-dir", type=Path, default=Path("labels"))
    p.add_argument("--weak-labels", type=Path, default=None,
                   help="Recording-level weak_labels.csv. AUDIO_FILE_ID must match recording_id.")
    p.add_argument("--output-dir", type=Path, default=Path("dataset"))

    p.add_argument("--window", type=float, default=3.0)
    p.add_argument("--strong-max-duration", type=float, default=5.0,
                   help="Annotations <= this duration are strong events.")
    p.add_argument("--candidate-stride", type=float, default=3.0)
    p.add_argument("--candidate-overlap", type=float, default=0.0,
                   help="Maximum allowed overlap (seconds) with strong events for weak candidates.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    # Spatial holdout is intentionally not a build-time operation.
    # Build all sites once; create spatial folds afterward from site_id.
    return p.parse_args()


def parse_label_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = re.split(r"\t+|\s{2,}", line)
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                print(f"WARNING: {path}:{line_number}: cannot parse: {line}")
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                print(f"WARNING: {path}:{line_number}: invalid times: {line}")
                continue

            label = parts[2].strip()
            if end <= start:
                print(f"WARNING: {path}:{line_number}: end <= start: {line}")
                continue

            rows.append({
                "start": start,
                "end": end,
                "duration": end - start,
                "label": label,
                "line": line_number,
            })
    return rows


def collapse_quality_label(label: str):
    """Return (biological_species, recording_quality)."""
    label = str(label).strip()
    parts = label.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in QUALITY_CODES:
        return parts[0], parts[1]
    return label, ""


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def make_centered_window(event_start, event_end, duration, window):
    event_center = (event_start + event_end) / 2.0
    start = event_center - window / 2.0
    if start < 0:
        start = 0.0
    if start + window > duration:
        start = max(0.0, duration - window)
    return start, start + window


def make_strong_segments(audio_path, annotations, window, strong_max_duration):
    """Create one centered window per strong event, merging overlapping labels."""
    duration = sf.info(audio_path).duration
    strong = [a for a in annotations if a["duration"] <= strong_max_duration]
    windows = []

    for a in strong:
        if duration < window:
            continue
        start, end = make_centered_window(a["start"], a["end"], duration, window)

        species = set()
        qualities = set()
        events = []
        for b in strong:
            ov = overlap(start, end, b["start"], b["end"])
            if ov > 0:
                sp, q = collapse_quality_label(b["label"])
                species.add(sp)
                if q:
                    qualities.add(q)
                events.append((b, ov))

        windows.append({
            "start": start,
            "end": end,
            "species": sorted(species),
            "qualities": sorted(qualities),
            "event_start": a["start"],
            "event_end": a["end"],
            "event_label": a["label"],
            "n_events": len(events),
        })

    unique = {}
    for w in windows:
        key = (round(w["start"], 4), round(w["end"], 4))
        if key not in unique:
            unique[key] = w
        else:
            unique[key]["species"] = sorted(set(unique[key]["species"]) | set(w["species"]))
            unique[key]["qualities"] = sorted(set(unique[key]["qualities"]) | set(w["qualities"]))
            unique[key]["n_events"] = max(unique[key]["n_events"], w["n_events"])

    return list(unique.values())


def make_candidate_windows(audio_path, annotations, weak_recording_labels,
                            window, stride, candidate_overlap):
    """Make unlabeled 3-s candidates from a weak-labeled recording."""
    duration = sf.info(audio_path).duration
    if duration < window:
        return []

    strong = [a for a in annotations if a["duration"] <= 5.0]
    candidates = []
    start = 0.0

    while start + window <= duration + 1e-9:
        end = start + window
        strong_overlap = sum(
            overlap(start, end, a["start"], a["end"]) for a in strong
        )
        if strong_overlap <= candidate_overlap:
            candidates.append({
                "start": start,
                "end": end,
                "weak_recording_labels": weak_recording_labels,
            })
        start += stride
    return candidates


def write_segment(audio_path, output_path, start, duration):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(audio_path) as src:
        samplerate = src.samplerate
        channels = src.channels
        frames = int(round(duration * samplerate))
        src.seek(int(round(start * samplerate)))
        data = src.read(frames, dtype="float32")

        if len(data) < frames:
            import numpy as np
            shape = (frames - len(data),) if channels == 1 else (frames - len(data), channels)
            data = np.concatenate([data, np.zeros(shape, dtype=data.dtype)], axis=0)
        sf.write(output_path, data, samplerate)


def parse_recording_metadata(recording_id):
    match = re.match(r"^(.+?)_(\d{8})_(\d{6})(?:_.+)?$", recording_id)
    if not match:
        return "UNKNOWN", "", ""
    site_id, date_str, time_str = match.groups()
    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return site_id, dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except ValueError:
        return site_id, "", ""


def recording_split(recording_id, seed, train_frac, val_frac):
    """Assign a recording to random train/val/test reproducibly.

    This split is independent of site. The site_id is retained separately so
    that spatial folds can be created later without rebuilding segments.
    """
    digest = hashlib.md5(f"{seed}:{recording_id}".encode("utf-8")).hexdigest()
    x = int(digest[:8], 16) / 0xFFFFFFFF
    if x < train_frac:
        return "train"
    if x < train_frac + val_frac:
        return "val"
    return "test"


def load_weak_labels(path: Path):
    """Load and validate recording-level weak activity labels."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Weak-label CSV has no header: {path}")
        fields = list(reader.fieldnames)
        if "AUDIO_FILE_ID" not in fields:
            raise ValueError("Weak-label CSV must contain AUDIO_FILE_ID")
        species_columns = [c for c in fields if c.startswith("SPECIES_")]
        if not species_columns:
            raise ValueError("Weak-label CSV contains no SPECIES_* columns")

        rows = {}
        duplicate_ids = []
        bad_values = []
        for row in reader:
            rid = (row.get("AUDIO_FILE_ID") or "").strip()
            if not rid:
                continue
            if rid in rows:
                duplicate_ids.append(rid)
                continue
            clean = dict(row)
            for col in species_columns:
                raw = (clean.get(col) or "").strip()
                # Preserve the original string value. Validate numeric 0-3 when nonblank.
                if raw:
                    try:
                        value = float(raw)
                        if value not in {0.0, 1.0, 2.0, 3.0}:
                            bad_values.append((rid, col, raw))
                    except ValueError:
                        bad_values.append((rid, col, raw))
            rows[rid] = clean

    if duplicate_ids:
        raise ValueError(f"Duplicate AUDIO_FILE_ID values in weak labels: {duplicate_ids[:10]}")
    if bad_values:
        example = ", ".join(map(str, bad_values[:5]))
        raise ValueError(f"Weak labels contain non-0/1/2/3 values; examples: {example}")

    return fields, species_columns, rows


def weak_activity_pairs(row, species_columns):
    pairs = []
    presence = []
    for col in species_columns:
        raw = (row.get(col) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            code = col[len("SPECIES_"):]
            pairs.append(f"{code}:{raw}")
            presence.append(code)
    return "|".join(pairs), "|".join(presence)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not (0 < args.train_frac < 1):
        raise SystemExit("--train-frac must be between 0 and 1")
    if not (0 <= args.val_frac < 1):
        raise SystemExit("--val-frac must be >= 0 and < 1")
    if args.train_frac + args.val_frac >= 1:
        raise SystemExit("--train-frac + --val-frac must be < 1")

    strong_dir = args.output_dir / "strong"
    candidate_dir = args.output_dir / "candidates"
    strong_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    weak_fields = []
    weak_species_columns = []
    weak_rows = {}
    if args.weak_labels:
        weak_fields, weak_species_columns, weak_rows = load_weak_labels(args.weak_labels)
        print(f"Loaded weak labels: {len(weak_rows)} recordings, {len(weak_species_columns)} species columns")

    audio_files = sorted(
        p for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )
    if not audio_files:
        raise SystemExit(f"No audio files found in {args.audio_dir}")

    metadata_rows = []
    recording_rows = []
    used_strong_species = set()
    weak_matched = set()
    counts = Counter()

    for audio_path in audio_files:
        recording_id = audio_path.stem
        site_id, recording_date, recording_time = parse_recording_metadata(recording_id)
        # Labels mirror the audio directory hierarchy by site.
        #
        # Example:
        #   audio_root/
        #       INCT4/recording.wav
        #       INCT17/recording.wav
        #
        #   label_root/
        #       INCT4/recording.txt
        #       INCT17/recording.txt
        #
        # The relative path is preserved, with only the audio extension
        # replaced by .txt. This also supports deeper nesting below each site.
        relative_audio = audio_path.relative_to(args.audio_dir)
        label_path = (args.label_dir / relative_audio).with_suffix(".txt")

        # Backward-compatible fallback for a flat label directory.
        if not label_path.exists():
            flat_label_path = args.label_dir / f"{recording_id}.txt"
            if flat_label_path.exists():
                label_path = flat_label_path

        annotations = parse_label_file(label_path) if label_path.exists() else []
        if not annotations:
            print(f"SKIP: no valid label file for {audio_path}")
            continue

        weak_row = weak_rows.get(recording_id)
        if weak_row is not None:
            weak_matched.add(recording_id)
            weak_activity, weak_presence = weak_activity_pairs(weak_row, weak_species_columns)
        else:
            weak_activity, weak_presence = "", ""

        split = recording_split(
            recording_id, args.seed, args.train_frac, args.val_frac
        )
        info = sf.info(audio_path)
        n_strong = sum(a["duration"] <= args.strong_max_duration for a in annotations)
        n_long = sum(a["duration"] > args.strong_max_duration for a in annotations)

        recording_rows.append({
            "recording_id": recording_id,
            "site_id": site_id,
            "recording_date": recording_date,
            "recording_time": recording_time,
            "audio_file": audio_path.name,
            "duration": info.duration,
            "n_annotations": len(annotations),
            "n_strong_annotations": n_strong,
            "n_long_annotations": n_long,
            "has_strong_labels": int(n_strong > 0),
            "has_weak_labels": int(weak_row is not None),
            "weak_recording_labels": weak_activity,
            "weak_recording_presence_labels": weak_presence,
            "split": split,
        })

        strong_segments = make_strong_segments(
            audio_path, annotations, args.window, args.strong_max_duration
        )

        for seg in strong_segments:
            used_strong_species.update(seg["species"])
            label_string = ";".join(seg["species"])
            quality_string = ";".join(seg["qualities"])
            safe_labels = "_".join(seg["species"]) or "UNLABELED"
            segment_id = (
                f"{recording_id}_{seg['start']:.3f}_{seg['end']:.3f}_{safe_labels}"
            )
            output_file = strong_dir / f"{segment_id}.wav"
            write_segment(audio_path, output_file, seg["start"], args.window)

            metadata_rows.append({
                "segment_id": segment_id,
                "recording_id": recording_id,
                "site_id": site_id,
                "recording_date": recording_date,
                "recording_time": recording_time,
                "audio_file": audio_path.name,
                "segment_file": str(output_file.relative_to(args.output_dir)),
                "split": split,
                "start_time": round(seg["start"], 6),
                "end_time": round(seg["end"], 6),
                "duration": args.window,
                "source_type": "strong_event",
                "strong_species_labels": label_string,
                "n_strong_species": len(seg["species"]),
                "strong_quality_labels": quality_string,
                "n_strong_events_in_window": seg["n_events"],
                "event_start": seg["event_start"],
                "event_end": seg["event_end"],
                "event_label": seg["event_label"],
                "weak_recording_labels": weak_activity,
                "weak_recording_presence_labels": weak_presence,
                "segment_target_status": "strong_multilabel",
            })
            counts["strong_event_segments"] += 1

        # Weak candidates come from the explicit recording-level weak table.
        # They remain unlabeled at the segment level.
        if weak_row is not None:
            candidate_segments = make_candidate_windows(
                audio_path,
                annotations,
                weak_activity,
                args.window,
                args.candidate_stride,
                args.candidate_overlap,
            )
            for seg in candidate_segments:
                segment_id = (
                    f"{recording_id}_{seg['start']:.3f}_{seg['end']:.3f}_candidate"
                )
                output_file = candidate_dir / f"{segment_id}.wav"
                write_segment(audio_path, output_file, seg["start"], args.window)
                metadata_rows.append({
                    "segment_id": segment_id,
                    "recording_id": recording_id,
                    "site_id": site_id,
                    "recording_date": recording_date,
                    "recording_time": recording_time,
                    "audio_file": audio_path.name,
                    "segment_file": str(output_file.relative_to(args.output_dir)),
                    "split": split,
                    "start_time": round(seg["start"], 6),
                    "end_time": round(seg["end"], 6),
                    "duration": args.window,
                    "source_type": "weak_candidate",
                    "strong_species_labels": "",
                    "n_strong_species": 0,
                    "strong_quality_labels": "",
                    "n_strong_events_in_window": 0,
                    "event_start": "",
                    "event_end": "",
                    "event_label": "",
                    "weak_recording_labels": weak_activity,
                    "weak_recording_presence_labels": weak_presence,
                    "segment_target_status": "unknown_from_weak_recording",
                })
                counts["weak_candidate_segments"] += 1

        counts[f"recordings_{split}"] += 1
        print(
            f"{recording_id}: annotations={len(annotations)} "
            f"strong_clips={len(strong_segments)} "
            f"weak={'yes' if weak_row is not None else 'no'} "
            f"split={split}"
        )

    # Preserve the weak-label table exactly as supplied, while adding parsed recording metadata.
    if args.weak_labels:
        weak_output_rows = []
        for rid, row in weak_rows.items():
            site_id, date, time = parse_recording_metadata(rid)
            out = dict(row)
            out["recording_id"] = rid
            out["site_id"] = site_id
            out["recording_date"] = date
            out["recording_time"] = time
            weak_output_rows.append(out)
        weak_out_fields = ["recording_id", "site_id", "recording_date", "recording_time"] + weak_fields
        write_csv(args.output_dir / "recording_weak_labels.csv", weak_output_rows, weak_out_fields)

    metadata_fields = [
        "segment_id", "recording_id", "site_id", "recording_date", "recording_time",
        "audio_file", "segment_file", "split", "start_time", "end_time", "duration",
        "source_type", "strong_species_labels", "n_strong_species", "strong_quality_labels",
        "n_strong_events_in_window", "event_start", "event_end", "event_label",
        "weak_recording_labels", "weak_recording_presence_labels", "segment_target_status",
    ]
    write_csv(args.output_dir / "metadata.csv", metadata_rows, metadata_fields)

    recording_fields = [
        "recording_id", "site_id", "recording_date", "recording_time", "audio_file",
        "duration", "n_annotations", "n_strong_annotations", "n_long_annotations",
        "has_strong_labels", "has_weak_labels", "weak_recording_labels",
        "weak_recording_presence_labels", "split",
    ]
    write_csv(args.output_dir / "recording_splits.csv", recording_rows, recording_fields)

    # Site inventory is generated from the audio actually discovered.
    # No site names are hard-coded, so this works for any number of sites.
    site_counts = Counter(r["site_id"] for r in recording_rows)
    site_inventory_rows = [
        {"site_id": site_id, "n_recordings": site_counts[site_id]}
        for site_id in sorted(site_counts)
    ]
    write_csv(
        args.output_dir / "site_inventory.csv",
        site_inventory_rows,
        ["site_id", "n_recordings"],
    )

    # Species vocabulary = union of weak-label species and strong biological species.
    weak_species = {c[len("SPECIES_"):] for c in weak_species_columns}
    all_species = sorted(weak_species | used_strong_species)
    vocab_rows = []
    for i, species in enumerate(all_species):
        vocab_rows.append({
            "class_index": i,
            "species_code": species,
            "weak_label_column": f"SPECIES_{species}" if species in weak_species else "",
            "present_in_weak_table": int(species in weak_species),
            "present_in_strong_annotations": int(species in used_strong_species),
        })
    write_csv(
        args.output_dir / "species_vocabulary.csv",
        vocab_rows,
        ["class_index", "species_code", "weak_label_column", "present_in_weak_table", "present_in_strong_annotations"],
    )

    unmatched_weak = sorted(set(weak_rows) - weak_matched)
    missing_label_files = [
        r["recording_id"] for r in recording_rows
        if r["n_annotations"] == 0 and not r["has_weak_labels"]
    ]

    summary = [
        "Anuran multi-label segment build",
        "================================",
        f"Audio files discovered: {len(audio_files)}",
        f"Recordings included: {len(recording_rows)}",
        f"Metadata segments: {len(metadata_rows)}",
        f"Strong event segments: {counts['strong_event_segments']}",
        f"Weak candidate segments: {counts['weak_candidate_segments']}",
        f"Weak recordings loaded: {len(weak_rows)}",
        f"Weak recordings matched to audio: {len(weak_matched)}",
        f"Weak recordings without matching audio: {len(unmatched_weak)}",
        f"Strong species vocabulary: {len(used_strong_species)}",
        f"Final species vocabulary (union): {len(all_species)}",
        f"Sites discovered: {len(site_inventory_rows)}",
        f"Site IDs: {', '.join(row["site_id"] for row in site_inventory_rows)}",
        "",
        "Weak-label semantics: values are preserved as recording-level 0/1/2/3 calling activity.",
        "Weak values are not copied into segment-level targets.",
        "Strong _L/_M/_H suffixes are retained only as recording-quality metadata.",
        "Strong biological targets are multi-label species sets.",
        "",
    ]
    if unmatched_weak:
        summary.append("Unmatched weak AUDIO_FILE_ID examples: " + ", ".join(unmatched_weak[:20]))
    if missing_label_files:
        summary.append("Audio with neither strong annotation file nor weak row: " + ", ".join(missing_label_files[:20]))

    (args.output_dir / "build_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\nDONE")
    print(f"Metadata:             {args.output_dir / 'metadata.csv'}")
    print(f"Recording splits:     {args.output_dir / 'recording_splits.csv'}")
    if args.weak_labels:
        print(f"Weak recording table: {args.output_dir / 'recording_weak_labels.csv'}")
    print(f"Species vocabulary:   {args.output_dir / 'species_vocabulary.csv'}")
    print(f"Site inventory:       {args.output_dir / 'site_inventory.csv'}")
    print(f"Strong clips:         {strong_dir}")
    print(f"Weak candidates:      {candidate_dir}")
    print("\nStrong target design: multi-label species; weak labels remain recording-level ordinal activity.")
    print("Random train/val/test splitting is by recording.")
    print("Spatial holdouts are NOT created during building; use site_id in metadata.csv to create them afterward.")


if __name__ == "__main__":
    main()
