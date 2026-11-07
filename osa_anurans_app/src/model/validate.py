"""Checks which prevent silent embedding or train/test misalignment."""

from .dataloader import EmbeddingStore
from .datasets import DatasetSplit, TrainingDataset


def validate_training_data(dataset: TrainingDataset, split: DatasetSplit) -> int:
    """Check selected examples before creating a classifier.

    Checks:
    - Nonempty aligned embedding rows and labels.
    - No recording shared between train and test.
    - Readable finite vectors with a consistent feature count.

    Returns:
    - Embedding dimension used to construct the probe's input layer.
    """
    if not dataset.rows or len(dataset.rows) != len(dataset.labels):
        raise ValueError("No aligned single-label examples available for training")
    train_recordings = {dataset.rows[i].recording_id for i in split.train}
    test_recordings = {dataset.rows[i].recording_id for i in split.test}
    if train_recordings & test_recordings:
        raise ValueError("A recording appears in both training and test partitions")

    store = EmbeddingStore()
    dimension = None
    for row in dataset.rows:
        vector = store.read(row)
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError(f"Inconsistent embedding dimensions for window {row.window_id}")
    return dimension
