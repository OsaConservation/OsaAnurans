"""Species-specific review priority; lower scores are reviewed first."""

import math


def similarity_fraction(positive_recording_count: int, balance: float = 25) -> float:
    """Allocate more similarity matches when positive recording coverage is small."""
    if type(positive_recording_count) is not int or positive_recording_count < 0:
        raise ValueError("positive_recording_count must be a nonnegative integer")
    if isinstance(balance, bool) or not math.isfinite(balance) or balance <= 0:
        raise ValueError("similarity balance must be finite and positive")
    return balance / (balance + positive_recording_count)


def score_candidate(probability: float, context: dict | None = None) -> float:
    """Distance from 50%, a heuristic for the initial softmax classifier.

    Rules:
    - Return abs(2*p - 1): zero at p=0.5, one at p=0 or p=1.
    - Smaller scores receive annotation priority.
    - Require a finite probability between zero and one.

    Limitations:
    - Multiclass softmax output is not independent species-presence confidence.
    - This heuristic is uncalibrated; it is not overall multiclass entropy.
    - context is unused for now; later factors may include annotation counts.
    """
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Candidate probability must be finite and between zero and one")
    return abs(2 * probability - 1)
