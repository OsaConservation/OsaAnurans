"""Stable recording/window identities and lossless model-input audio files."""

import hashlib
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import soundfile as sf

from .records import RecordingRecord, WindowRecord

RECORDING_NAMESPACE = UUID("b32da944-57c2-4f0e-a1ce-7e85b8849e20")
WINDOW_NAMESPACE = UUID("9185a058-e8bd-4b53-bfba-c57f5fb7d862")
WINDOW_POLICY = "5s-stride5-wrap-v1"
SAMPLE_RATE = 32_000
WINDOW_SAMPLES = SAMPLE_RATE * 5


def file_hash(path: Path) -> str:
    """Compute SHA-256 in 1 MiB blocks without reading the whole file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identify_recording(path: Path) -> RecordingRecord:
    """Read source metadata and derive a stable recording ID.

    Steps:
    1. Read frame count/sample rate; reject empty audio.
    2. Hash file bytes and derive a UUID from that hash.
    3. Return source metadata and the resolved local path.
    """
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Empty or invalid recording: {path}")
    digest = file_hash(path)
    return RecordingRecord(
        str(uuid5(RECORDING_NAMESPACE, digest)),
        path.resolve(),
        digest,
        info.samplerate,
        info.frames,
    )


def recording_windows(recording: RecordingRecord) -> list[WindowRecord]:
    """Create five-second window records in source sample positions.

    Rules:
    - Start a new window every five seconds; no overlap.
    - Derive IDs from recording ID, sample bounds, and extraction policy.
    - Keep the final five-second logical bound even if audio needs padding.
    - Describe windows here; decoding and padding happen in bacpipe.
    """
    length = 5 * recording.sample_rate
    windows = []
    for start in range(0, recording.frames, length):
        # The final window still spans five seconds; bacpipe wraps its short tail.
        end = start + length
        key = f"{recording.recording_id}:{start}:{end}:{WINDOW_POLICY}"
        windows.append(
            WindowRecord(
                str(uuid5(WINDOW_NAMESPACE, key)),
                recording.recording_id,
                start,
                end,
                recording.sample_rate,
                WINDOW_POLICY,
            )
        )
    return windows


def valid_window_audio(path: Path) -> bool:
    """Check the WAV header: five seconds, mono, 32 kHz, FLOAT samples.

    Returns:
    - False for missing/unreadable files or a different format.
    - This checks metadata, not equality with a newly prepared waveform.
    """
    try:
        info = sf.info(path)
        return (
            info.frames == WINDOW_SAMPLES
            and info.samplerate == SAMPLE_RATE
            and info.channels == 1
            and info.subtype == "FLOAT"
        )
    except (OSError, RuntimeError):
        return False


def write_window_audio(path: Path, samples: np.ndarray) -> None:
    """Save the exact model-input waveform as a local FLOAT WAV.

    Steps:
    1. Require 160000 finite mono samples.
    2. If a valid WAV exists, require identical samples before reusing it.
    3. Otherwise write a temporary WAV and rename it to the final path.

    Unhappy paths:
    - Changed samples under an existing valid window file raise an error.
    - Review/version the extraction policy before changing window audio.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.shape != (WINDOW_SAMPLES,) or not np.isfinite(samples).all():
        raise ValueError("Window audio must contain exactly 160000 finite mono samples")
    if valid_window_audio(path):
        existing, rate = sf.read(path, dtype="float32")
        if rate != SAMPLE_RATE or not np.array_equal(existing, samples):
            raise ValueError(
                f"Window waveform changed under an existing identity: {path}. "
                "Review preprocessing and version the window policy before replacing audio."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.wav")
    sf.write(temporary, samples, SAMPLE_RATE, subtype="FLOAT")
    temporary.replace(path)
