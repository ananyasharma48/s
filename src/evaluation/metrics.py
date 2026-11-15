"""Binary classification metrics for cell-state evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probability: np.ndarray) -> dict[str, float]:
    """Compute primary and secondary metrics with class 1 as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_probability)),
        "auprc": float(average_precision_score(y_true, y_probability)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
