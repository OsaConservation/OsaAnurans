"""Create-only window audio uploads with content verification."""

import base64
from pathlib import Path

from ..config import AppConfig
from ..records import WindowRecord


def file_checksum(path: Path) -> str:
    """Compute the file's base64 CRC32C checksum in blocks for GCS verification."""
    import google_crc32c

    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return base64.b64encode(checksum.digest()).decode("ascii")


class StorageStore:
    """Upload standalone window WAVs without overwriting existing objects.

    Rules:
    - Object name: <configured prefix>/<window_id>.wav.
    - Reuse an existing object only when its size/checksum match local audio.
    - Never upload vectors or delete previously uploaded audio here.
    """

    def __init__(self, config: AppConfig, client=None):
        """Use a supplied test client or create a GCS client for the configured project."""
        if client is None:
            from google.cloud import storage

            client = storage.Client(project=config.project)
        self.client = client
        self.config = config

    def check_access(self) -> None:
        """Check metadata access without creating or modifying objects."""
        self.client.get_bucket(self.config.bucket)

    def upload_window(self, window: WindowRecord, path: Path) -> str:
        """Create or verify one window's GCS object and return its gs:// URI.

        Steps:
        1. Require a local file and a window ID safe for one path component.
        2. Compute the local checksum and attempt a create-only upload.
        3. Reload object metadata and verify size/checksum, including on retries.

        Unhappy paths:
        - A conflicting object raises an error; it is never overwritten.
        """
        from google.api_core.exceptions import PreconditionFailed

        if not path.is_file():
            raise FileNotFoundError(f"Window audio not found: {path}")
        if not window.window_id or "/" in window.window_id or "\\" in window.window_id:
            raise ValueError("A window ID must be a single nonempty path component")

        name = f"{self.config.prefix.rstrip('/')}/{window.window_id}.wav"
        blob = self.client.bucket(self.config.bucket).blob(name)
        checksum = file_checksum(path)
        try:
            blob.upload_from_filename(
                str(path),
                content_type="audio/wav",
                if_generation_match=0,
                checksum="crc32c",
            )
        except PreconditionFailed:
            # An earlier successful attempt or another process may own this name.
            pass

        blob.reload()
        if blob.size != path.stat().st_size or blob.crc32c != checksum:
            raise ValueError(f"Existing/uploaded GCS audio differs from local window: {name}")
        return f"gs://{self.config.bucket}/{name}"
