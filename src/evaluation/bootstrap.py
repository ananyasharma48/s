"""Raw-file-group bootstrap and paired fold-permutation utilities."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_metrics


DEFAULT_METRICS = ("balanced_accuracy", "macro_f1", "auroc", "auprc")


def group_bootstrap_metrics(
    predictions: pd.DataFrame,
    group_column: str,
    replicates: int,
    seed: int,
    metric_names: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Resample complete groups and calculate metrics for every valid replicate."""
    required = {"y_true", "y_pred", "probability_sc_u", group_column}
    if missing := required - set(predictions.columns):
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    groups = predictions[group_column].drop_duplicates().tolist()
    if len(groups) < 2:
        raise ValueError("group bootstrap requires at least two groups")
    group_positions = {
        group: np.flatnonzero(predictions[group_column].eq(group).to_numpy())
        for group in groups
    }
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_pred"].to_numpy(dtype=int)
    probabilities = predictions["probability_sc_u"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    records = []
    attempts = 0
    maximum_attempts = replicates * 20
    while len(records) < replicates and attempts < maximum_attempts:
        attempts += 1
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        positions = np.concatenate([group_positions[group] for group in sampled_groups])
        selected_truth = y_true[positions]
        if np.unique(selected_truth).size != 2:
            continue
        metrics = binary_metrics(selected_truth, y_pred[positions], probabilities[positions])
        records.append({
            "bootstrap_replicate": len(records),
            **{name: metrics[name] for name in metric_names},
        })
    if len(records) != replicates:
        raise RuntimeError(f"only {len(records)} valid bootstrap replicates were generated")
    return pd.DataFrame(records)


def percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    """Return a two-sided percentile interval."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(values, dtype=float), [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def paired_group_bootstrap_differences(
    reference: pd.DataFrame,
    comparator: pd.DataFrame,
    group_column: str,
    replicates: int,
    seed: int,
    metric_names: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Bootstrap paired groups and return comparator-minus-reference differences."""
    columns = ["sample_id", "y_true", "y_pred", "probability_sc_u", group_column]
    reference_selected = reference[columns].rename(columns={
        "y_pred": "reference_prediction",
        "probability_sc_u": "reference_probability",
    })
    comparator_selected = comparator[columns[:-1]].rename(columns={
        "y_true": "comparator_truth",
        "y_pred": "comparator_prediction",
        "probability_sc_u": "comparator_probability",
    })
    paired = reference_selected.merge(
        comparator_selected, on="sample_id", how="inner", validate="one_to_one"
    )
    if len(paired) != len(reference) or len(paired) != len(comparator):
        raise ValueError("paired models do not cover identical samples")
    if not np.array_equal(paired["y_true"], paired["comparator_truth"]):
        raise ValueError("paired models disagree on truth labels")
    groups = paired[group_column].drop_duplicates().tolist()
    group_positions = {
        group: np.flatnonzero(paired[group_column].eq(group).to_numpy())
        for group in groups
    }
    rng = np.random.default_rng(seed)
    records = []
    while len(records) < replicates:
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        positions = np.concatenate([group_positions[group] for group in sampled_groups])
        truth = paired["y_true"].to_numpy(dtype=int)[positions]
        if np.unique(truth).size != 2:
            continue
        reference_metrics = binary_metrics(
            truth,
            paired["reference_prediction"].to_numpy(dtype=int)[positions],
            paired["reference_probability"].to_numpy(dtype=float)[positions],
        )
        comparator_metrics = binary_metrics(
            truth,
            paired["comparator_prediction"].to_numpy(dtype=int)[positions],
            paired["comparator_probability"].to_numpy(dtype=float)[positions],
        )
        records.append({
            "bootstrap_replicate": len(records),
            **{name: comparator_metrics[name] - reference_metrics[name] for name in metric_names},
        })
    return pd.DataFrame(records)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided paired sign-flip test for a small number of fold differences."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or differences.size < 1 or not np.isfinite(differences).all():
        raise ValueError("differences must be a nonempty finite vector")
    if differences.size > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 pairs")
    observed = abs(float(differences.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        statistics.append(abs(float((differences * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))
