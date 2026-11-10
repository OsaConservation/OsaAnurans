"""Small immutable records shared by pipeline stages."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordingRecord:
    """Identify a source recording and describe its original audio format.

    Fields:
    - recording_id: Stable UUID from content_hash; stored on each source window.
    - path: Resolved local file location, not an identity.
    - content_hash: SHA-256 of file bytes; derives the ID and checks catalog metadata.
    - sample_rate: Original sample frames per second, before resampling.
    - frames: Total source frames. A stereo frame has one sample per channel.

    Identity rules:
    - Moves/renames preserve identity; identical copies share it.
    - Changed bytes get a new ID so old labels/vectors cannot attach to new audio.
    - Re-encoding or embedded metadata changes also change the hash.

    Usage:
    - Duration in seconds = frames / sample_rate.
    - Describes the source; stores no decoded samples.

    Example:
        recording = identify_recording(Path("recording.wav"))  # from audio.py
        duration_seconds = recording.frames / recording.sample_rate
        windows = recording_windows(recording)

    """

    recording_id: str
    path: Path
    content_hash: str
    sample_rate: int
    frames: int


@dataclass(frozen=True)
class WindowRecord:
    """Identify one window and locate it in its source recording.

    Fields:
    - window_id: UUID from recording_id, sample bounds, and policy.
    - recording_id: Identity of the source recording.
    - start_sample: Inclusive source frame position, not seconds.
    - end_sample: Exclusive source frame position, not seconds.
    - sample_rate: Original source sample frames per second.
    - policy: Extraction version; default makes 5s windows starting every 5s.

    Usage:
    - Use window_id to find the same window's WAV, vector, candidates, and labels.
    - Divide sample bounds by sample_rate for source seconds.
    - Cloud media belongs to candidates, not window identity.

    Example:
        window = WindowRecord("window-id", "recording-id", 220500, 441000, 44100)
        assert window.start_seconds == 5.0
        assert window.end_seconds == 10.0

    """

    window_id: str
    recording_id: str
    start_sample: int
    end_sample: int
    sample_rate: int
    policy: str = "5s-stride5-wrap-v1"

    @property
    def start_seconds(self) -> float:
        """Convert the inclusive source sample position to source seconds."""
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        """Convert the exclusive source sample position to source seconds."""
        return self.end_sample / self.sample_rate


@dataclass(frozen=True)
class EmbeddingRow:
    """Index one window's embedding without storing the vector in this record.

    Purpose:
    - Look up a window ID in SQLite to find its vector's file and row number.
    - Keep vectors in a recording-level .npy matrix, not in this record.
    - Matrix rows are windows; columns are embedding features.

    Fields:
    - window_id: Window that produced the vector; shared by its labels/candidates.
    - recording_id: Source identity; keeps its windows together in train/test splits.
    - config_id: Identifier for the embedding model/preprocessing configuration.
    - path: Local .npy matrix path, not an audio file or individual vector file.
    - row_index: Zero-based row position in that matrix.

    Configuration rules:
    - config_id covers checkpoint checksum, bacpipe version, sample rate,
      window policy, and resampling-library versions.
    - It is not a classifier training run ID.
    - One window may have vectors from multiple configurations.
    - (window_id, config_id) identifies its embedding in the catalog.
    - Reject vectors whose config_id differs from the classifier's saved config_id.

    Usage:
    - EmbeddingStore.read(row) handles loading and validation in application code.
    - Vectors stay local.

    Example:
        row = EmbeddingRow(
            "window-id", "recording-id", "perch-config-id",
            Path("embeddings/recording-id.npy"), 2,
        )
        matrix = np.load(row.path, mmap_mode="r", allow_pickle=False)
        vector = matrix[row.row_index]  # Third window's vector; Perch has 1536 features.

    """

    window_id: str
    recording_id: str
    config_id: str
    path: Path
    row_index: int


@dataclass(frozen=True)
class IntervalLabel:
    """Represent one known-presence interval from the initial annotation CSV.

    Fields:
    - recording_id: Source identity, resolved from the CSV's audiofilename.
    - species_id: Species known to be present during the interval.
    - start: Interval start in source seconds, not sample positions.
    - end: Interval end in source seconds; exclusive for overlap calculations.

    Label rules:
    - Any positive-duration window overlap establishes presence.
    - Partial coverage is accepted with a warning.
    - Touching a boundary alone is not overlap.
    - Outside intervals remains unknown, not absent; CSV labels are not exhaustive.

    Example:
        interval = IntervalLabel("recording-id", "DENEBR", 5.0, 10.0)
        labels = map_intervals_to_windows([interval], windows)  # from annotations.py

    """

    recording_id: str
    species_id: str
    start: float
    end: float


@dataclass(frozen=True)
class LabelCounts:
    """Store vote counts and derive consensus for one window/species pair.

    Fields:
    - window_id: Window being labeled; used to look up its WAV and embedding row.
    - species_id: Species whose presence is being judged.
    - present: Present vote count; defaults to zero.
    - absent: Absent vote count; defaults to zero.

    Voting rules:
    - Count every present/absent answer; unsure contributes neither vote.
    - No latest-answer selection or annotator deduplication.
    - Strict majority wins; ties (including zero/zero) return None for unknown.
    - Different species can independently be present in one window.

    Usage:
    - Save counts in consensus.csv so the majority can be checked later.
    - CSV bootstrap presence uses present=1, not an actual annotator vote.
    - Initial training uses windows with exactly one known-present species.

    Example:
        label = LabelCounts("window-id", "DENEBR", present=3, absent=1)
        assert label.response == "present"
        assert LabelCounts("window-id", "DENEBR", 1, 1).response is None

    """

    window_id: str
    species_id: str
    present: int = 0
    absent: int = 0

    @property
    def response(self) -> str | None:
        """Return the majority response, or None when present/absent counts tie."""
        if self.present == self.absent:
            return None
        return "present" if self.present > self.absent else "absent"
