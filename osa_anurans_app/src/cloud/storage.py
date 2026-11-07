"""Create-only window audio uploads with content verification."""

import base64
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from ..config import AppConfig
from ..records import WindowRecord

logger = logging.getLogger(__name__)


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
    - Delete only canonical window WAVs explicitly selected by cleanup.
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

    @contextmanager
    def publication_lock(self):
        """Serialize cloud uploads/cleanup across computers using a create-only object.

        - No automatic expiry: a slow publisher must not lose its lock.
        - After a crash, remove the lock only after confirming the process stopped.
        """
        from google.api_core.exceptions import GoogleAPICallError, NotFound, PreconditionFailed

        name = f"{self.config.prefix.rstrip('/')}/.publication.lock"
        blob = self.client.bucket(self.config.bucket).blob(name)
        owner = json.dumps({"owner": uuid4().hex, "started_at": datetime.now(UTC).isoformat()})
        try:
            blob.upload_from_string(owner, content_type="application/json", if_generation_match=0)
        except PreconditionFailed as exc:
            raise RuntimeError(
                "Another cloud upload/cleanup holds the publication lock. "
                f"If the previous process stopped, remove gs://{self.config.bucket}/{name} before retrying."
            ) from exc
        except GoogleAPICallError as exc:
            raise RuntimeError(f"Unable to acquire the cloud publication lock: {exc}") from exc
        generation = int(blob.generation)
        release = True
        try:
            yield
        except (KeyboardInterrupt, SystemExit):
            # A submitted BigQuery job can keep running after its local wait stops.
            release = False
            logger.warning(
                "Cloud operation interrupted; retaining %s until outstanding cloud jobs are checked",
                name,
            )
            raise
        finally:
            if release:
                try:
                    blob.delete(if_generation_match=generation)
                except NotFound:
                    pass

    def list_window_audio(self) -> dict:
        """List clip metadata once; no audio downloads or individual existence requests.

        - Keys are flat window filenames without .wav.
        - Ignore nested objects, lock files, and non-WAV files.
        - Cleanup applies the stricter UUID check before deletion.
        """
        prefix = f"{self.config.prefix.rstrip('/')}/"
        clips = {}
        for blob in self.client.list_blobs(self.config.bucket, prefix=prefix):
            relative = blob.name.removeprefix(prefix)
            if blob.name.startswith(prefix) and relative.endswith(".wav") and "/" not in relative:
                clips[relative[:-4]] = blob
        return clips

    def storage_usage(self) -> dict[str, int]:
        """Sum bucket metadata, including references and retained soft-deleted objects.

        - No object data is downloaded.
        - Report active and soft-deleted bytes separately: cleanup does not purge retention.
        """
        active = sum(blob.size or 0 for blob in self.client.list_blobs(self.config.bucket))
        deleted = sum(
            blob.size or 0 for blob in self.client.list_blobs(self.config.bucket, soft_deleted=True)
        )
        return {"active_bytes": active, "soft_deleted_bytes": deleted}

    def delete_window_audio(self, window_id: str, blob) -> bool:
        """Delete exactly the listed generation of a canonical window WAV.

        - Preserve unexpected names, other prefixes, and other buckets.
        - An already-missing object is a successful retry, with no bytes removed.
        """
        from google.api_core.exceptions import NotFound

        try:
            canonical = str(UUID(window_id)) == window_id
        except ValueError:
            canonical = False
        if not canonical:
            raise ValueError(f"Not a canonical window ID: {window_id}")
        expected = f"{self.config.prefix.rstrip('/')}/{window_id}.wav"
        if blob.name != expected or blob.bucket.name != self.config.bucket:
            raise ValueError("Refusing to delete an object outside the window audio path")
        try:
            blob.delete(if_generation_match=int(blob.generation))
        except NotFound:
            return False
        return True

    def upload_window(self, window: WindowRecord, path: Path) -> str:
        """Create or verify one window's GCS object and return its gs:// URI.

        Steps:
        1. Require a local file and a window ID safe for one path component.
        2. Compute the local checksum and attempt a create-only upload.
        3. Reload object metadata and verify size/checksum, including on retries.

        Unhappy paths:
        - A conflicting object raises an error; it is never overwritten.
        """

        if not path.is_file():
            raise FileNotFoundError(f"Window audio not found: {path}")
        if not window.window_id or "/" in window.window_id or "\\" in window.window_id:
            raise ValueError("A window ID must be a single nonempty path component")

        name = f"{self.config.prefix.rstrip('/')}/{window.window_id}.wav"
        return self._upload_audio(name, path)

    def upload_reference(self, species_id: str, reference_id: str, path: Path) -> str:
        """Create/verify reference audio outside the window cleanup prefix."""
        if not species_id.isalnum() or str(UUID(reference_id)) != reference_id:
            raise ValueError("Invalid species/reference identity")
        return self._upload_audio(f"references/{species_id}/{reference_id}.wav", path)

    def download_reference(self, uri: str, path: Path) -> None:
        """Read existing reference audio to verify content before reusing its row."""
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected a GCS reference URI: {uri}")
        bucket, separator, name = uri[5:].partition("/")
        if not bucket or not separator or not name:
            raise ValueError(f"Invalid GCS reference URI: {uri}")
        self.client.bucket(bucket).blob(name).download_to_filename(str(path))

    def _upload_audio(self, name: str, path: Path) -> str:
        """Shared create-only WAV upload; matching retries preserve existing bytes."""
        from google.api_core.exceptions import PreconditionFailed

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
