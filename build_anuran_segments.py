#!/usr/bin/env python3
"""
Build a 3-second anuran audio-segment dataset from annotation files of the form:

START<TAB>END<TAB>LABEL

Example:
0.823375    1.082150    ADEMAR_M
2.223113    2.599513    ADEMAR_M

Assumptions tailored to the supplied annotations:
- Short intervals are treated as strong/event annotations.
- Very long intervals are treated as weak/presence annotations and are NOT
  automatically used as strong positive labels.
- Strong events produce 3-s clips centered on the event.
- Weakly labeled recordings can optionally produce an unlabeled candidate pool.
- A clip may have multiple labels if strong events overlap it.
- Splitting is by recording, not by segment, to prevent leakage.

Directory layout expected:
project/
  audio/
    INCT4_20191023_181500.wav
    ...
  labels/
    INCT4_20191023_181500.txt
    ...
  build_dataset.py

Outputs:
  dataset/
    strong/
      *.wav
    candidates/
      *.wav                  (optional)
    metadata.csv
    recording_splits.csv
"""

from pathlib import Path
import argparse
import csv
import hashlib
import random
import re

import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, default=Path("audio"))
    p.add_argument("--label-dir", type=Path, default=Path("labels"))
    p.add_argument("--output-dir", type=Path, default=Path("dataset"))

    p.add_argument("--window", type=float, default=3.0)
    p.add_argument("--strong-max-duration", type=float, default=5.0,
                   help="Annotations <= this duration are treated as strong events.")
    p.add_argument("--strong-center", action="store_true", default=True)
    p.add_argument("--candidate-stride", type=float, default=3.0)
    p.add_argument("--candidate-overlap", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def parse_label_file(path):
    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = re.split(r"\t+|\s{2,}", line)

            if len(parts) < 3:
                # Fall back to any whitespace separation.
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


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def make_centered_window(event_start, event_end, duration, window):
    """Center the 3-s window on the event; clamp to recording boundaries."""
    event_center = (event_start + event_end) / 2.0
    start = event_center - window / 2.0

    if start < 0:
        start = 0.0

    if start + window > duration:
        start = max(0.0, duration - window)

    end = start + window
    return start, end


def make_strong_segments(audio_path, annotations, window, strong_max_duration):
    """
    One candidate clip per strong annotation.
    If several strong events fall in the same 3-s window, their labels are
    merged rather than producing duplicate audio files.
    """
    info = sf.info(audio_path)
    duration = info.duration

    strong = [
        a for a in annotations
        if a["duration"] <= strong_max_duration
    ]

    windows = []

    for a in strong:
        if duration < window:
            continue

        start, end = make_centered_window(
            a["start"], a["end"], duration, window
        )

        # Find every strong event that occurs in this same clip.
        labels = set()
        events = []

        for b in strong:
            ov = overlap(start, end, b["start"], b["end"])
            if ov > 0:
                labels.add(b["label"])
                events.append((b, ov))

        windows.append({
            "start": start,
            "end": end,
            "labels": sorted(labels),
            "source_type": "strong_event",
            "event_start": a["start"],
            "event_end": a["end"],
            "event_label": a["label"],
            "n_events": len(events),
        })

    # Deduplicate windows that are effectively identical.
    unique = {}
    for w in windows:
        key = (
            round(w["start"], 4),
            round(w["end"], 4),
        )
        if key not in unique:
            unique[key] = w
        else:
            unique[key]["labels"] = sorted(
                set(unique[key]["labels"]) | set(w["labels"])
            )
            unique[key]["n_events"] = max(
                unique[key]["n_events"], w["n_events"]
            )

    return list(unique.values())


def make_candidate_windows(audio_path, annotations, window, stride,
                            candidate_overlap):
    """
    Candidate pool from recordings containing long/weak annotations.

    These clips are deliberately NOT assigned the weak annotation as a
    strong class label. The metadata records which weak labels occur in
    the parent recording.
    """
    info = sf.info(audio_path)
    duration = info.duration

    if duration < window:
        return []

    weak = [a for a in annotations if a["duration"] > 5.0]

    if not weak:
        return []

    weak_labels = sorted(set(a["label"] for a in weak))

    candidates = []
    start = 0.0

    while start + window <= duration + 1e-9:
        end = start + window

        strong_overlap = 0.0
        for a in annotations:
            if a["duration"] <= 5.0:
                strong_overlap += overlap(
                    start, end, a["start"], a["end"]
                )

        if strong_overlap <= candidate_overlap:
            candidates.append({
                "start": start,
                "end": end,
                "labels": [],
                "weak_recording_labels": weak_labels,
                "source_type": "weak_candidate",
                "event_start": "",
                "event_end": "",
                "event_label": "",
                "n_events": 0,
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

        # Normally the source is long enough because windows are clamped.
        # Pad only as a safeguard.
        if len(data) < frames:
            import numpy as np
            if channels == 1:
                pad = np.zeros(frames - len(data), dtype=data.dtype)
            else:
                pad = np.zeros(
                    (frames - len(data), channels),
                    dtype=data.dtype
                )
            data = np.concatenate([data, pad], axis=0)

        sf.write(output_path, data, samplerate)


def recording_split(recording_id, seed, train_frac, val_frac):
    """
    Deterministic split by recording ID.
    This prevents nearly identical overlapping segments from leaking
    between train/validation/test.
    """
    digest = hashlib.md5(
        f"{seed}:{recording_id}".encode("utf-8")
    ).hexdigest()

    x = int(digest[:8], 16) / 0xFFFFFFFF

    if x < train_frac:
        return "train"
    elif x < train_frac + val_frac:
        return "val"
    return "test"


def main():
    args = parse_args()

    strong_dir = args.output_dir / "strong"
    candidate_dir = args.output_dir / "candidates"

    strong_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        p for p in args.audio_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    )

    if not audio_files:
        raise SystemExit(f"No audio files found in {args.audio_dir}")

    metadata_rows = []
    recording_rows = []

    for audio_path in audio_files:
        label_path = args.label_dir / f"{audio_path.stem}.txt"

        if not label_path.exists():
            print(f"SKIP: no label file for {audio_path.name}")
            continue

        annotations = parse_label_file(label_path)

        if not annotations:
            print(f"SKIP: no valid annotations in {label_path.name}")
            continue

        info = sf.info(audio_path)
        recording_id = audio_path.stem

        split = recording_split(
            recording_id,
            args.seed,
            args.train_frac,
            args.val_frac,
        )

        n_strong = sum(
            a["duration"] <= args.strong_max_duration
            for a in annotations
        )
        n_weak = sum(
            a["duration"] > args.strong_max_duration
            for a in annotations
        )

        recording_rows.append({
            "recording_id": recording_id,
            "audio_file": audio_path.name,
            "duration": info.duration,
            "n_annotations": len(annotations),
            "n_strong_annotations": n_strong,
            "n_weak_annotations": n_weak,
            "split": split,
        })

        strong_segments = make_strong_segments(
            audio_path,
            annotations,
            args.window,
            args.strong_max_duration,
        )

        candidate_segments = make_candidate_windows(
            audio_path,
            annotations,
            args.window,
            args.candidate_stride,
            args.candidate_overlap,
        )

        print(
            f"{audio_path.name}: "
            f"{len(annotations)} annotations -> "
            f"{len(strong_segments)} strong clips, "
            f"{len(candidate_segments)} candidate clips"
        )

        # Write strong clips.
        for i, seg in enumerate(strong_segments):
            label_string = ";".join(seg["labels"])
            safe_labels = "_".join(seg["labels"]) or "UNLABELED"

            segment_id = (
                f"{recording_id}_"
                f"{seg['start']:.3f}_"
                f"{seg['end']:.3f}_"
                f"{safe_labels}"
            )

            output_file = strong_dir / f"{segment_id}.wav"

            write_segment(
                audio_path,
                output_file,
                seg["start"],
                args.window,
            )

            metadata_rows.append({
                "segment_id": segment_id,
                "recording_id": recording_id,
                "audio_file": audio_path.name,
                "segment_file": str(output_file.relative_to(args.output_dir)),
                "split": split,
                "start_time": round(seg["start"], 6),
                "end_time": round(seg["end"], 6),
                "duration": args.window,
                "source_type": "strong_event",
                "labels": label_string,
                "n_labels": len(seg["labels"]),
                "event_start": seg["event_start"],
                "event_end": seg["event_end"],
                "event_label": seg["event_label"],
                "n_events_in_window": seg["n_events"],
                "weak_recording_labels": "",
            })

        # Write candidate clips.
        for seg in candidate_segments:
            segment_id = (
                f"{recording_id}_"
                f"{seg['start']:.3f}_"
                f"{seg['end']:.3f}_candidate"
            )

            output_file = candidate_dir / f"{segment_id}.wav"

            write_segment(
                audio_path,
                output_file,
                seg["start"],
                args.window,
            )

            metadata_rows.append({
                "segment_id": segment_id,
                "recording_id": recording_id,
                "audio_file": audio_path.name,
                "segment_file": str(output_file.relative_to(args.output_dir)),
                "split": split,
                "start_time": round(seg["start"], 6),
                "end_time": round(seg["end"], 6),
                "duration": args.window,
                "source_type": "weak_candidate",
                "labels": "",
                "n_labels": 0,
                "event_start": "",
                "event_end": "",
                "event_label": "",
                "n_events_in_window": 0,
                "weak_recording_labels": ";".join(
                    seg["weak_recording_labels"]
                ),
            })

    metadata_path = args.output_dir / "metadata.csv"

    fields = [
        "segment_id",
        "recording_id",
        "audio_file",
        "segment_file",
        "split",
        "start_time",
        "end_time",
        "duration",
        "source_type",
        "labels",
        "n_labels",
        "event_start",
        "event_end",
        "event_label",
        "n_events_in_window",
        "weak_recording_labels",
    ]

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata_rows)

    split_path = args.output_dir / "recording_splits.csv"

    with split_path.open("w", newline="", encoding="utf-8") as f:
        fields2 = [
            "recording_id",
            "audio_file",
            "duration",
            "n_annotations",
            "n_strong_annotations",
            "n_weak_annotations",
            "split",
        ]
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(recording_rows)

    print()
    print("DONE")
    print(f"Metadata: {metadata_path}")
    print(f"Splits:   {split_path}")
    print(f"Strong clips:    {strong_dir}")
    print(f"Candidate clips: {candidate_dir}")
    print()
    print("Important: train/val/test are split by RECORDING, not by clip.")


if __name__ == "__main__":
    main()
