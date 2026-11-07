"""Load explicit paths and validate configuration before doing work."""

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Hold input paths, local output paths, and model/cloud settings.

    Groups:
    - audio_roots: Directories to ingest new recordings from.
    - annotations_csv: Optional CSV labels for one-time seed imports.
    - artifacts: Store SQLite, WAVs, vectors, snapshots, and training runs.
    - species_manifest: JSON input for adding species references and seed votes.
    - project/dataset/bucket/prefix/location: Explicit cloud destinations.
    - device/seed/epochs/batch_size/learning_rate/test_fraction: Training settings.
    - candidates_per_species: Maximum saved candidates per species; default 100.
    - similarity_balance: Positive recording count giving a 50/50 candidate mix.
    """

    audio_roots: tuple[Path, ...]
    annotations_csv: Path | None
    artifacts: Path
    project: str = "osaanurans"
    dataset: str = "osafrogdataset"
    bucket: str = "osa-frog-data"
    prefix: str = "windows/v1"
    location: str = "us-south1"
    device: str = "auto"
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.0005
    test_fraction: float = 0.2
    candidates_per_species: int = 100
    similarity_balance: float = 25
    species_manifest: Path | None = None

    @property
    def species_manifest_path(self) -> Path:
        """Use the configured manifest, or inputs/species.json beside the output folder."""
        return self.species_manifest or self.artifacts.parent / "inputs" / "species.json"


def default_config(directory: Path | None = None) -> Path:
    """Prefer personal settings; fall back to the supplied example config."""
    directory = directory if directory is not None else Path(__file__).parents[1]
    local = directory / "config.local.toml"
    return local if local.is_file() else directory / "config.example.toml"


def load_config(path: Path) -> AppConfig:
    """Read TOML settings and return a validated AppConfig.

    Steps:
    1. Read paths, training, candidates, and cloud sections.
    2. Resolve relative paths against the TOML directory, not the working directory.
    3. Apply dataclass defaults and validate values.

    Boundaries:
    - Source files need not exist for cached training or cloud operations.
    - Import checks source availability separately.
    """
    path = path.resolve()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = data.get("paths", {})

    def resolve(value: str) -> Path:
        """Expand ~ and resolve relative paths from the configuration directory."""
        item = Path(value).expanduser()
        return (path.parent / item).resolve() if not item.is_absolute() else item.resolve()

    config = AppConfig(
        audio_roots=tuple(resolve(p) for p in paths["audio_roots"]),
        annotations_csv=resolve(paths["annotations_csv"]) if paths.get("annotations_csv") else None,
        artifacts=resolve(paths.get("artifacts", "outputs")),
        species_manifest=resolve(paths.get("species_manifest", "inputs/species.json")),
        **data.get("cloud", {}),
        **data.get("training", {}),
        **data.get("candidates", {}),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Check paths/settings syntax without opening source recordings.

    Checks:
    - Valid BigQuery identifiers and bucket/object-prefix format.
    - Supported device and positive training/candidate limits.
    - Test fraction strictly between zero and one.
    """
    if not config.audio_roots:
        raise ValueError("At least one audio root must be configured")
    for name in (config.project, config.dataset):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid BigQuery project/dataset: {name}")
    if not config.bucket or "/" in config.bucket:
        raise ValueError("bucket must be a bucket name, without gs:// or a path")
    if not config.prefix or config.prefix.startswith("/") or ".." in config.prefix.split("/"):
        raise ValueError("prefix must be a relative GCS object prefix")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if (
        config.epochs < 1
        or config.batch_size < 1
        or not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
    ):
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if not 0 < config.test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if type(config.candidates_per_species) is not int or config.candidates_per_species < 1:
        raise ValueError("candidates_per_species must be a positive integer")
    if (
        isinstance(config.similarity_balance, bool)
        or not math.isfinite(config.similarity_balance)
        or config.similarity_balance <= 0
    ):
        raise ValueError("similarity_balance must be finite and positive")


def validate_import_sources(config: AppConfig) -> None:
    """Only audio/CSV imports require the original source paths to be available."""
    if any(not path.is_dir() for path in config.audio_roots):
        raise ValueError("Every configured audio root must be an existing directory.")
    if config.annotations_csv is not None and not config.annotations_csv.is_file():
        raise ValueError(f"Annotation CSV not found: {config.annotations_csv}")
