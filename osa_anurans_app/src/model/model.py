"""Artifact metadata and inference around bacpipe's linear probe."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from bacpipe.embedding_evaluation.probing.train_probe import LinearProbe

from ..files import write_json


def choose_device(requested: str) -> str:
    """Choose where PyTorch runs the probe.

    Rules:
    - auto: Use CUDA if PyTorch can access it, otherwise CPU.
    - cuda: Fail when CUDA is unavailable.
    - cpu: Always use CPU.
    - Perch's ONNX provider is selected separately in bacpipe_adapter.py.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested for probe training but PyTorch cannot access a GPU")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported probe device: {requested}")
    return requested


@dataclass
class ProbeArtifact:
    """Keep a trained bacpipe probe with the metadata needed to interpret it.

    Fields:
    - model: Linear layer producing one logit per trained species.
    - species_ids: Species order matching output columns.
    - config_id: Embedding configuration accepted by this classifier.
    - run_id: Training run identity, also used when generating candidate IDs.
    - device: CPU/CUDA location of the model and prediction inputs.
    """

    model: LinearProbe
    species_ids: tuple[str, ...]
    config_id: str
    run_id: str
    device: str = "cpu"

    @property
    def embedding_dimension(self) -> int:
        """Return the number of features required in each input vector."""
        return self.model.probe.in_features

    def predict_logits(self, embeddings: np.ndarray) -> np.ndarray:
        """Return raw class scores for a matrix of embedding vectors.

        Steps:
        1. Check input dimensions and finite values.
        2. Run the saved model without gradients on its selected device.
        3. Return finite NumPy scores on CPU, in species_ids column order.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dimension:
            raise ValueError("Embedding shape does not match the saved probe")
        if not np.isfinite(embeddings).all():
            raise ValueError("Prediction embeddings must be finite")
        self.model.eval()
        with torch.inference_mode():
            inputs = torch.from_numpy(embeddings).to(self.device)
            logits = self.model(inputs)
        result = logits.cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Probe produced nonfinite logits")
        return result

    def predict_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply softmax to raw scores; each row sums to one across species.

        Notes:
        - Subtract the largest logit before exponentiation to avoid overflow.
        - Outputs are competing class probabilities, not independent presences.
        """
        logits = self.predict_logits(embeddings)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def save_artifact(artifact: ProbeArtifact, directory: Path) -> None:
    """Write model weights and interpretation metadata to a run directory.

    Files:
    - model.pt: PyTorch parameter state, not a pickled model object.
    - species.json: Species order matching the output columns.
    - model.json: Format/task, input dimension, embedding config, and run ID.

    Write rule:
    - Replace each file after its temporary write completes; not one transaction
      across all files in the directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "model.pt.tmp"
    torch.save(artifact.model.state_dict(), temporary)
    temporary.replace(directory / "model.pt")
    write_json(directory / "species.json", list(artifact.species_ids))
    write_json(
        directory / "model.json",
        {
            "format_version": 1,
            "task": "single_label_multiclass",
            "embedding_config_id": artifact.config_id,
            "embedding_dimension": artifact.embedding_dimension,
            "species_ids": list(artifact.species_ids),
            "run_id": artifact.run_id,
        },
    )


def load_artifact(directory: Path, device: str = "cpu") -> ProbeArtifact:
    """Rebuild a bacpipe probe from saved metadata and weights.

    Steps:
    1. Check format/task and unique species order with at least two classes.
    2. Construct the matching linear layer on the requested device.
    3. Load weights only and switch the model to evaluation mode.

    Returns:
    - ProbeArtifact ready to predict without retraining or reading source audio.
    """
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("task") != "single_label_multiclass":
        raise ValueError("Unsupported probe artifact format")
    species = tuple(metadata["species_ids"])
    if len(species) < 2 or len(set(species)) != len(species):
        raise ValueError("Invalid artifact species mapping")
    device = choose_device(device)
    model = LinearProbe(metadata["embedding_dimension"], len(species)).to(device)
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return ProbeArtifact(
        model, species, metadata["embedding_config_id"], metadata["run_id"], device
    )
