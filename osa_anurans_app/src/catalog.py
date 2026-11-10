"""Local window identities and explicit mappings to embedding array rows."""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import soundfile as sf

from .records import EmbeddingRow, RecordingRecord, WindowRecord


class Catalog:
    """Index local recordings, windows, vector locations, and upload progress.

    Tables:
    - recordings: Content identities and original audio metadata.
    - recording_paths: Current path-to-recording associations.
    - windows: Source bounds, local WAV path, GCS URI, and registration flag.
    - window_embeddings: (window_id, config_id) -> NumPy file and row index.

    Usage:
    - Open with a context manager so the SQLite connection closes afterward.
    - Store vectors/audio in files, not SQLite blobs.
    - Validate file/window alignment before committing their mappings.
    """

    def __init__(self, path: Path):
        """Open/create the catalog and its tables with foreign keys enabled."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id TEXT PRIMARY KEY, path TEXT NOT NULL,
                content_hash TEXT NOT NULL, sample_rate INTEGER NOT NULL,
                frames INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS windows (
                window_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL REFERENCES recordings,
                start_sample INTEGER NOT NULL, end_sample INTEGER NOT NULL,
                sample_rate INTEGER NOT NULL, policy TEXT NOT NULL,
                audio_path TEXT,
                registered INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS window_embeddings (
                window_id TEXT NOT NULL REFERENCES windows,
                config_id TEXT NOT NULL, path TEXT NOT NULL, row_index INTEGER NOT NULL,
                PRIMARY KEY (window_id, config_id),
                UNIQUE (config_id, path, row_index)
            );
            CREATE TABLE IF NOT EXISTS recording_paths (
                path TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL REFERENCES recordings
            );
            CREATE INDEX IF NOT EXISTS windows_by_recording
                ON windows(recording_id, start_sample);
            INSERT OR IGNORE INTO recording_paths(path, recording_id)
                SELECT path, recording_id FROM recordings;
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def add_recording(self, recording: RecordingRecord) -> None:
        """Store source metadata and associate its current path with its ID.

        Rules:
        - Reusing an ID requires the same hash, sample rate, and frame count.
        - Identical contents at different paths share the recording row.
        - Replaced file contents update the path association, preserving old windows.
        """
        if recording.sample_rate <= 0 or recording.frames <= 0:
            raise ValueError("A recording needs a positive sample rate and frame count")
        path = str(recording.path.resolve())
        with self.connection:
            existing = self.connection.execute(
                "SELECT content_hash, sample_rate, frames FROM recordings WHERE recording_id=?",
                (recording.recording_id,),
            ).fetchone()
            metadata = (recording.content_hash, recording.sample_rate, recording.frames)
            if existing and tuple(existing) != metadata:
                raise ValueError(
                    f"Recording identity has conflicting metadata: {recording.recording_id}"
                )
            self.connection.execute(
                "INSERT INTO recordings VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(recording_id) DO UPDATE SET path=excluded.path",
                (
                    recording.recording_id,
                    path,
                    recording.content_hash,
                    recording.sample_rate,
                    recording.frames,
                ),
            )
            # A changed file gets a new content identity; its previous windows remain intact.
            self.connection.execute(
                "INSERT INTO recording_paths VALUES (?, ?) ON CONFLICT(path) "
                "DO UPDATE SET recording_id=excluded.recording_id",
                (path, recording.recording_id),
            )

    def _validate_artifact(
        self,
        windows: list[WindowRecord],
        config_id: str,
        path: Path,
        audio_paths: list[Path],
    ) -> None:
        """Validate files and source bounds before writing catalog mappings.

        Checks:
        - One recording, unique windows, and one audio path per embedding row.
        - A real finite matrix with a nonzero feature count and matching row count.
        - Five-second mono WAVs and valid original-recording sample bounds.
        - Existing window IDs describe the same recording/bounds/policy.
        """
        if not config_id:
            raise ValueError("An embedding configuration ID is required")
        if len(windows) != len(audio_paths) or not windows:
            raise ValueError("Each embedding row needs a window and local audio file")
        if len({window.window_id for window in windows}) != len(windows):
            raise ValueError("An embedding artifact contains duplicate window IDs")
        if len({window.recording_id for window in windows}) != 1:
            raise ValueError("Each embedding artifact must belong to one recording")

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if array.ndim != 2 or array.shape[0] != len(windows) or array.shape[1] == 0:
                raise ValueError("Embedding array shape does not match its window mapping")
            if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
                array.dtype, np.complexfloating
            ):
                raise ValueError("Embedding array must contain real numeric values")
            for start in range(0, len(array), 4096):
                if not np.isfinite(array[start : start + 4096]).all():
                    raise ValueError("Embedding array contains nonfinite values")
        finally:
            if isinstance(array, np.memmap):
                array._mmap.close()

        for window, audio_path in zip(windows, audio_paths, strict=True):
            recording = self.connection.execute(
                "SELECT sample_rate, frames FROM recordings WHERE recording_id=?",
                (window.recording_id,),
            ).fetchone()
            if recording is None:
                raise ValueError(f"Window has no registered recording: {window.window_id}")
            if (
                window.sample_rate != recording["sample_rate"]
                or window.start_sample < 0
                or window.start_sample >= recording["frames"]
                or window.end_sample - window.start_sample != 5 * window.sample_rate
            ):
                raise ValueError(f"Invalid five-second window bounds: {window.window_id}")

            info = sf.info(audio_path)
            if info.channels != 1 or info.samplerate <= 0 or info.frames != info.samplerate * 5:
                raise ValueError(f"Window audio must be five-second mono audio: {audio_path}")

            existing = self.connection.execute(
                "SELECT recording_id, start_sample, end_sample, sample_rate, policy "
                "FROM windows WHERE window_id=?",
                (window.window_id,),
            ).fetchone()
            metadata = (
                window.recording_id,
                window.start_sample,
                window.end_sample,
                window.sample_rate,
                window.policy,
            )
            if existing and tuple(existing) != metadata:
                raise ValueError(f"Window identity has conflicting metadata: {window.window_id}")

    def register_artifact(
        self,
        windows: list[WindowRecord],
        config_id: str,
        path: Path,
        audio_paths: list[Path],
    ) -> None:
        """Save complete window/audio/vector mappings in one SQLite transaction.

        Steps:
        1. Validate the already-written files and source metadata.
        2. Store each window's local audio path.
        3. Store its embedding file/row under (window_id, config_id).

        Rules:
        - Row order must match window/audio order.
        - Existing metadata registration state is preserved.
        """
        self._validate_artifact(windows, config_id, path, audio_paths)
        path = path.resolve()
        with self.connection:
            # Release the selected rows together so a repaired/reordered mapping
            # cannot collide with another selected window's previous location.
            self.connection.executemany(
                "DELETE FROM window_embeddings WHERE window_id=? AND config_id=?",
                ((window.window_id, config_id) for window in windows),
            )
            for index, (window, audio_path) in enumerate(zip(windows, audio_paths)):
                self.connection.execute(
                    "INSERT INTO windows (window_id, recording_id, start_sample, end_sample, "
                    "sample_rate, policy, audio_path) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(window_id) DO UPDATE SET audio_path=excluded.audio_path",
                    (
                        window.window_id,
                        window.recording_id,
                        window.start_sample,
                        window.end_sample,
                        window.sample_rate,
                        window.policy,
                        str(audio_path.resolve()),
                    ),
                )
                self.connection.execute(
                    "INSERT INTO window_embeddings VALUES (?, ?, ?, ?)",
                    (window.window_id, config_id, str(path), index),
                )

    def recordings(self) -> list[RecordingRecord]:
        """Return recordings with a current path association, one path per ID."""
        # Select an active alias instead of a stale canonical path after file replacement.
        query = (
            "SELECT r.*, MIN(p.path) AS active_path FROM recordings r "
            "JOIN recording_paths p USING(recording_id) GROUP BY r.recording_id ORDER BY r.recording_id"
        )
        with closing(self.connection.execute(query)) as cursor:
            return [
                RecordingRecord(
                    r["recording_id"],
                    Path(r["active_path"]),
                    r["content_hash"],
                    r["sample_rate"],
                    r["frames"],
                )
                for r in cursor
            ]

    def stored_recordings(self) -> list[RecordingRecord]:
        """Return all recording metadata, including replaced source contents.

        Usage:
        - Publish older windows even if their source path now has different audio.
        - Treat stored paths as history; they may no longer contain these bytes.
        - Use recording_aliases() to resolve current source paths.
        """
        with closing(
            self.connection.execute("SELECT * FROM recordings ORDER BY recording_id")
        ) as cursor:
            return [
                RecordingRecord(
                    row["recording_id"],
                    Path(row["path"]),
                    row["content_hash"],
                    row["sample_rate"],
                    row["frames"],
                )
                for row in cursor
            ]

    def recording_aliases(self) -> list[RecordingRecord]:
        """Return each current path association, including copies sharing one ID."""
        query = (
            "SELECT r.*, p.path AS alias_path FROM recordings r "
            "JOIN recording_paths p USING(recording_id) ORDER BY p.path"
        )
        with closing(self.connection.execute(query)) as cursor:
            return [
                RecordingRecord(
                    r["recording_id"],
                    Path(r["alias_path"]),
                    r["content_hash"],
                    r["sample_rate"],
                    r["frames"],
                )
                for r in cursor
            ]

    @staticmethod
    def _window(row) -> WindowRecord:
        """Read WindowRecord fields from a SQLite row, ignoring catalog-only columns."""
        return WindowRecord(**{key: row[key] for key in WindowRecord.__dataclass_fields__})

    def windows(self) -> list[WindowRecord]:
        """Return all windows ordered by recording ID and original start sample."""
        rows = self.connection.execute("SELECT * FROM windows ORDER BY recording_id, start_sample")
        return [self._window(row) for row in rows]

    def embedding_rows(
        self, config_id: str, *, recording_id: str | None = None
    ) -> list[EmbeddingRow]:
        """Find vector file/row locations for one embedding configuration.

        Returns:
        - EmbeddingRow records ordered by recording ID and window start sample.
        - File contents stay on disk; this reads only SQLite metadata.
        - Optional recording_id limits the query to one recording.
        """
        query = (
            "SELECT e.*, w.recording_id FROM window_embeddings e "
            "JOIN windows w USING(window_id) WHERE config_id=?"
        )
        parameters = [config_id]
        if recording_id is not None:
            query += " AND w.recording_id=?"
            parameters.append(recording_id)
        query += " ORDER BY w.recording_id, w.start_sample"
        rows = self.connection.execute(query, parameters)
        return [
            EmbeddingRow(
                r["window_id"],
                r["recording_id"],
                r["config_id"],
                Path(r["path"]),
                r["row_index"],
            )
            for r in rows
        ]

    def config_ids(self) -> list[str]:
        """List embedding configurations with indexed vectors."""
        return [
            r[0]
            for r in self.connection.execute(
                "SELECT DISTINCT config_id FROM window_embeddings ORDER BY config_id"
            )
        ]

    def audio_path(self, window_id: str) -> Path | None:
        """Find a window's local WAV path; return None if no path is registered."""
        row = self.connection.execute(
            "SELECT audio_path FROM windows WHERE window_id=?", (window_id,)
        ).fetchone()
        return Path(row[0]) if row and row[0] else None

    def pending_windows(self) -> list[WindowRecord]:
        """List windows not yet marked registered in BigQuery.

        Notes:
        - May include windows whose WAV upload succeeded but metadata write failed.
        - Publication further filters this list to IDs in the candidate file.
        """
        rows = self.connection.execute(
            "SELECT * FROM windows WHERE registered=0 ORDER BY window_id"
        )
        return [self._window(row) for row in rows]

    def mark_registered(self, window_id: str) -> None:
        """Mark window metadata registered in BigQuery, independently of cloud audio."""
        with self.connection:
            updated = self.connection.execute(
                "UPDATE windows SET registered=1 WHERE window_id=?", (window_id,)
            )
            if not updated.rowcount:
                raise ValueError(f"Unknown window: {window_id}")
