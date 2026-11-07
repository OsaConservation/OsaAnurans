"""Train a small linear probe while enforcing a recording-disjoint split.

Responsibilities:
- Supply local vector rows and a recording-level train/test split.
- Use bacpipe's LinearProbe and train_linear_probe for model training.
- Save weights, species order, metrics, configuration, and split membership.
"""

import csv
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from bacpipe.embedding_evaluation.probing.train_probe import train_linear_probe
from torch.utils.data import DataLoader

from ..config import AppConfig
from ..files import write_json
from .dataloader import ProbeDataset
from .datasets import TrainingDataset, split_by_recording
from .evaluation import classification_metrics
from .model import LinearProbe, ProbeArtifact, choose_device, save_artifact
from .validate import validate_training_data

logger = logging.getLogger(__name__)


def _evaluate(artifact: ProbeArtifact, rows, labels, batch_size: int) -> dict:
    """Predict class indices in batches and calculate classification metrics."""
    dataset = ProbeDataset(rows, labels)
    predictions = []
    loss_sum = 0.0
    for embeddings, targets in DataLoader(dataset, batch_size=batch_size):
        logits = artifact.predict_logits(embeddings.numpy())
        predictions.extend(logits.argmax(axis=1).tolist())
        # Stable cross-entropy, averaged over examples rather than batches.
        scores = logits.astype(np.float64)
        losses = (
            np.logaddexp.reduce(scores, axis=1) - scores[np.arange(len(scores)), targets.numpy()]
        )
        loss_sum += float(losses.sum())
    metrics = classification_metrics(labels, predictions, artifact.species_ids)
    metrics["loss"] = loss_sum / len(labels) if labels else None
    return metrics


def report_test_results(metrics: dict) -> None:
    """Show held-out results separately from the trainer's epoch statistics."""
    print(flush=True)
    test = metrics["test"]
    if not test["examples"]:
        logger.info("Test set: unavailable; no held-out examples")
        return
    logger.info(
        "Test set: %s examples | Loss: %.4f | Accuracy: %.2f%%",
        test["examples"],
        test["loss"],
        100 * test["accuracy"],
    )
    logger.info("Species    Examples    Recall      F1")
    for species, values in test["per_class"].items():
        recall = f"{100 * values['recall']:.1f}%" if values["recall"] is not None else "n/a"
        f1 = f"{values['f1']:.3f}" if values["f1"] is not None else "n/a"
        logger.info("%-10s %8s %9s %7s", species, values["support"], recall, f1)


def _write_split(path: Path, dataset: TrainingDataset, split, species_ids: tuple[str, ...]) -> None:
    """Record which windows/species were assigned to each partition.

    Columns:
    - window_id/recording_id/species_id: Identify the selected example.
    - partition: train or test.
    - used: False when its species has no training support.
    """
    train_indices = set(split.train)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_id",
                "recording_id",
                "species_id",
                "partition",
                "used",
            ],
        )
        writer.writeheader()
        for index, (row, species) in enumerate(zip(dataset.rows, dataset.labels, strict=True)):
            writer.writerow(
                {
                    "window_id": row.window_id,
                    "recording_id": row.recording_id,
                    "species_id": species,
                    "partition": "train" if index in train_indices else "test",
                    "used": species in species_ids,
                }
            )


def _configuration_metadata(config: AppConfig) -> dict:
    """Convert settings to JSON-ready values, including path strings."""
    metadata = asdict(config)
    metadata["audio_roots"] = [str(path) for path in config.audio_roots]
    metadata["annotations_csv"] = str(config.annotations_csv) if config.annotations_csv else None
    metadata["artifacts"] = str(config.artifacts)
    return metadata


def train_classifier(
    dataset: TrainingDataset, output_dir: Path, config: AppConfig
) -> ProbeArtifact:
    """Train a single-label multiclass probe and save its evaluation evidence.

    Steps:
    1. Require a new/empty output directory and valid training settings.
    2. Split by recording and validate vector dimensions/values.
    3. Select species with training examples; require at least two.
    4. Train through bacpipe using our dataloader and seeded shuffle.
    5. Evaluate train/test predictions and save the model, metrics, and split.

    Rules:
    - Skip/report species absent from training, including their test examples.
    - Refuse to save a model containing nonfinite parameters.
    - Candidate selection/upload happens outside this function.

    Returns:
    - ProbeArtifact with trained weights and species/configuration metadata.
    """
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Run output must be a new or empty directory: {output_dir}")
    valid_rate = np.isfinite(config.learning_rate) and config.learning_rate > 0
    if config.epochs < 1 or config.batch_size < 1 or not valid_rate:
        raise ValueError("Invalid training settings")
    split = split_by_recording(dataset, config.test_fraction, config.seed)
    dimension = validate_training_data(dataset, split)
    species_ids = tuple(sorted({dataset.labels[i] for i in split.train}))
    skipped = sorted(set(dataset.species_ids) - set(species_ids))
    for species in skipped:
        logger.warning("Skipping species %s: no positive examples in training recordings", species)
    if len(species_ids) < 2:
        raise ValueError(
            "Need at least two species with positive examples in the training recordings"
        )

    # Use this same species-to-column order for training, metrics, and saved weights.
    label_to_index = {species: index for index, species in enumerate(species_ids)}
    train_rows = [dataset.rows[i] for i in split.train]
    train_labels = [label_to_index[dataset.labels[i]] for i in split.train]
    # The model cannot predict species for which it has no output column.
    test_indices = [i for i in split.test if dataset.labels[i] in label_to_index]
    test_rows = [dataset.rows[i] for i in test_indices]
    test_labels = [label_to_index[dataset.labels[i]] for i in test_indices]

    device = choose_device(config.device)
    logger.info("Linear probe device: %s; species: %s", device, ", ".join(species_ids))
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = LinearProbe(dimension, len(species_ids)).to(device)
    loader = DataLoader(
        ProbeDataset(train_rows, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    # Use bacpipe's trainer without its default window-level split/cache logic.
    model = train_linear_probe(
        model,
        loader,
        learning_rate=config.learning_rate,
        num_epochs=config.epochs,
        device=device,
    )
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Nonfinite probe parameters; classifier was not saved")

    artifact = ProbeArtifact(model, species_ids, dataset.config_id, output_dir.name, device)
    # Evaluate before saving so invalid predictions do not produce a completed model.
    metrics = {
        "task": "single_label_multiclass",
        "train": _evaluate(artifact, train_rows, train_labels, config.batch_size),
        "test": _evaluate(artifact, test_rows, test_labels, config.batch_size),
        "excluded": dataset.excluded,
        "test_examples_with_untrained_species": len(split.test) - len(test_indices),
        "held_out_evaluation_available": bool(test_indices),
        "seed": config.seed,
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    for species, values in metrics["test"]["per_class"].items():
        if test_indices and not values["support"]:
            logger.warning("No held-out positive examples for %s; recall/F1 unavailable", species)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_artifact(artifact, output_dir)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "skipped_species.json", skipped)
    write_json(output_dir / "config.json", _configuration_metadata(config))
    _write_split(output_dir / "split.csv", dataset, split, species_ids)
    report_test_results(metrics)
    return artifact
