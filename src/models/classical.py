"""Leakage-safe preprocessing and conventional baseline estimators."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Keep features observed in a minimum fraction of training samples."""

    def __init__(self, minimum_fraction: float = 0.05) -> None:
        self.minimum_fraction = minimum_fraction

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "PrevalenceFilter":
        values = np.asarray(x, dtype=float)
        self.support_ = np.mean(~np.isnan(values), axis=0) >= self.minimum_fraction
        if not np.any(self.support_):
            raise ValueError("prevalence filter removed every feature")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)[:, self.support_]


def fixed_vector_logistic(minimum_prevalence: float, seed: int) -> Pipeline:
    """Build a fixed-vector logistic pipeline whose transforms fit within CV."""
    return Pipeline([
        ("prevalence", PrevalenceFilter(minimum_prevalence)),
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=3000, solver="liblinear", random_state=seed)),
    ])


def peptide_random_forest(minimum_prevalence: float, seed: int, n_estimators: int, min_samples_leaf: int) -> Pipeline:
    """Build the fixed nonlinear peptide baseline."""
    return Pipeline([
        ("prevalence", PrevalenceFilter(minimum_prevalence)),
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )),
    ])


def batch_logistic(seed: int) -> Pipeline:
    """Build the diagnostic model that uses technical batch fields only."""
    return Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("model", LogisticRegression(class_weight="balanced", max_iter=3000, solver="liblinear", random_state=seed)),
    ])
