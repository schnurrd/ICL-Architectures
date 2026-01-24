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
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_ids = np.digitize(confidences, bin_edges[1:], right=True)
    
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(n_bins):
        mask = bin_ids == i
        
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            weight = np.sum(mask) / total_samples
            
            ece += weight * np.abs(bin_acc - bin_conf)
            
    return ece
