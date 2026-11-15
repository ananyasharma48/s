"""Probability-calibration metrics and reliability-bin summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def reliability_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize equal-width probability bins, including empty bins."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape != probabilities.shape or y_true.ndim != 1:
        raise ValueError("truth and probabilities must be same-length one-dimensional arrays")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and between zero and one")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), n_bins - 1)
    records = []
    for bin_index in range(n_bins):
        selected = assignments == bin_index
        count = int(selected.sum())
        records.append({
            "bin_index": bin_index,
            "bin_lower": float(edges[bin_index]),
            "bin_upper": float(edges[bin_index + 1]),
            "sample_count": count,
            "mean_probability": float(probabilities[selected].mean()) if count else float("nan"),
            "observed_positive_fraction": float(y_true[selected].mean()) if count else float("nan"),
        })
    return pd.DataFrame(records)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute count-weighted absolute reliability error."""
    table = reliability_table(y_true, probabilities, n_bins)
    occupied = table[table["sample_count"].gt(0)]
    error = (
        occupied["sample_count"]
        * (occupied["mean_probability"] - occupied["observed_positive_fraction"]).abs()
    ).sum() / occupied["sample_count"].sum()
    return float(error)


def calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Return uncalibrated Brier score and expected calibration error."""
    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities, n_bins),
    }
