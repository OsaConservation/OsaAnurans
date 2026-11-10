"""Evaluate only the single-label examples supported by the trained classes."""

import numpy as np


def classification_metrics(
    targets: list[int], predictions: list[int], species_ids: tuple[str, ...]
) -> dict:
    """Compare predicted/actual class indices using the saved species order.

    Outputs:
    - Confusion matrix: rows are actual classes; columns are predictions.
    - Overall accuracy and per-species support, precision, recall, and F1.

    Missing data:
    - No examples: accuracy is None.
    - No predicted examples for a class: precision is None.
    - No actual examples for a class: recall and F1 are None.
    """
    confusion = np.zeros((len(species_ids), len(species_ids)), dtype=int)
    for actual, predicted in zip(targets, predictions, strict=True):
        confusion[actual, predicted] += 1

    per_class = {}
    for index, species in enumerate(species_ids):
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        precision = true_positive / predicted_count if predicted_count else None
        recall = true_positive / support if support else None
        f1 = None
        if support:
            denominator = support + predicted_count
            f1 = 2 * true_positive / denominator
        per_class[species] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "examples": len(targets),
        "accuracy": float(np.trace(confusion) / len(targets)) if targets else None,
        "species_ids": list(species_ids),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }
