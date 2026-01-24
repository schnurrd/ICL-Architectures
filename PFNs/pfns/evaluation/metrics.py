from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1.0 - y_proba, y_proba])

    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(int)

    prob_true, prob_pred = calibration_curve(
        correct,
        confidences,
        n_bins=n_bins,
        strategy="uniform",
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:], right=True)
    bin_counts = np.bincount(bin_ids, minlength=n_bins)
    nonzero = bin_counts > 0
    weights = bin_counts[nonzero] / float(bin_counts.sum())
    return float(np.sum(weights * np.abs(prob_true - prob_pred)))
