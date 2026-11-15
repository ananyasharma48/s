"""Train and evaluate Stage 6 conventional baselines on immutable grouped folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from src.evaluation.metrics import binary_metrics
from src.models.classical import batch_logistic, fixed_vector_logistic, peptide_random_forest
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


MODEL_LABELS = {
    "majority": "Majority",
    "batch_logistic": "Batch logistic",
    "protein_logistic": "Protein logistic",
    "peptide_logistic": "Peptide logistic",
    "peptide_random_forest": "Peptide random forest",
}


def encode_labels(labels: pd.Series, positive_label: str) -> np.ndarray:
    """Encode the configured positive label as 1 and the other label as 0."""
    unique = set(labels)
    if positive_label not in unique or len(unique) != 2:
        raise ValueError(f"expected two labels including {positive_label}, found {sorted(unique)}")
    return labels.eq(positive_label).astype(int).to_numpy()


def predefined_inner_split(inner: pd.DataFrame, outer_fold: int, train_ids: list[str]) -> PredefinedSplit:
    """Convert saved inner validation assignments into a scikit-learn split."""
    validation = inner[(inner["outer_fold"] == outer_fold) & (inner["partition"] == "validation")]
    fold_by_sample = validation.set_index("sample_id")["inner_fold"]
    if set(train_ids) != set(fold_by_sample.index):
        raise ValueError(f"inner assignments do not cover outer fold {outer_fold} training samples")
    return PredefinedSplit(np.array([int(fold_by_sample.loc[sample_id]) for sample_id in train_ids]))


def prediction_rows(
    model_name: str,
    outer_fold: int,
    sample_ids: list[str],
    labels: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    positive_label: str,
) -> list[dict[str, object]]:
    """Create auditable per-sample prediction records."""
    return [
        {
            "model": model_name,
            "outer_fold": outer_fold,
            "sample_id": sample_id,
            "label_raw": label,
            "y_true": int(label == positive_label),
            "y_pred": int(prediction),
            "probability_sc_u": float(probability),
        }
        for sample_id, label, prediction, probability in zip(sample_ids, labels, predictions, probabilities)
    ]


def comparison_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot fold-level balanced accuracy and macro F1 for all baselines."""
    order = list(MODEL_LABELS)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, metric, title in zip(axes, ["balanced_accuracy", "macro_f1"], ["Balanced accuracy", "Macro F1"]):
        for position, model in enumerate(order):
            values = metrics.loc[metrics["model"].eq(model), metric].to_numpy()
            axis.scatter(np.full(len(values), position), values, alpha=0.75, s=34)
            axis.plot([position - 0.18, position + 0.18], [values.mean(), values.mean()], linewidth=2)
        axis.axhline(0.5, color="gray", linewidth=1, linestyle="--")
        axis.set_xticks(range(len(order)), [MODEL_LABELS[name] for name in order], rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(title + " across grouped outer folds")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_stage_six(config_path: str | Path = "configs/baseline.yaml") -> dict[str, object]:
    """Fit all baselines, save predictions/checkpoints, and summarize performance."""
    config = load_yaml_config(config_path)
    manifest = pd.read_parquet("data/interim/manifests/master_manifest.parquet")
    metadata = manifest.loc[manifest["is_valid_target"]].set_index("sample_id")
    peptide = pd.read_parquet("data/processed/fixed_features/peptide_by_sample.parquet")
    protein = pd.read_parquet("data/processed/fixed_features/protein_by_sample.parquet")
    outer = pd.read_csv("data/splits/outer_fold_assignments.csv")
    inner = pd.read_csv("data/splits/inner_fold_assignments.csv")
    positive_label = str(config["positive_label"])
    minimum_prevalence = float(config["minimum_feature_prevalence"])
    seed = int(config["seed"])
    c_grid = [float(value) for value in config["logistic_c_grid"]]
    batch_fields = ["batch_chromatography", "batch_sort", "batch_digest"]
    checkpoint_dir = Path("outputs/checkpoints/stage_06")
    table_dir = Path("outputs/tables")
    figure_dir = Path("outputs/figures")
    report_dir = Path("outputs/reports")
    for directory in (checkpoint_dir, table_dir, figure_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    all_predictions: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    tuning_records: list[dict[str, object]] = []
    for outer_fold in sorted(outer["outer_fold"].unique()):
        fold = outer[outer["outer_fold"] == outer_fold]
        train_ids = fold.loc[fold["partition"].eq("train"), "sample_id"].tolist()
        test_ids = fold.loc[fold["partition"].eq("test"), "sample_id"].tolist()
        train_labels = metadata.loc[train_ids, "label_raw"]
        test_labels = metadata.loc[test_ids, "label_raw"]
        y_train = encode_labels(train_labels, positive_label)
        y_test = encode_labels(test_labels, positive_label)
        inner_split = predefined_inner_split(inner, int(outer_fold), train_ids)
        models: dict[str, tuple[object, object, object]] = {}
        majority_probability = float(y_train.mean())
        majority_prediction = int(majority_probability >= 0.5)
        models["majority"] = (
            {"predicted_class": majority_prediction, "probability": majority_probability},
            np.full(len(test_ids), majority_prediction),
            np.full(len(test_ids), majority_probability),
        )
        batch_search = GridSearchCV(
            batch_logistic(seed),
            {"model__C": c_grid},
            scoring="balanced_accuracy",
            cv=inner_split,
            refit=True,
            n_jobs=1,
        )
        batch_search.fit(metadata.loc[train_ids, batch_fields], y_train)
        models["batch_logistic"] = (
            batch_search.best_estimator_,
            batch_search.predict(metadata.loc[test_ids, batch_fields]),
            batch_search.predict_proba(metadata.loc[test_ids, batch_fields])[:, 1],
        )
        for model_name, matrix in (("protein_logistic", protein), ("peptide_logistic", peptide)):
            search = GridSearchCV(
                fixed_vector_logistic(minimum_prevalence, seed),
                {"model__C": c_grid},
                scoring="balanced_accuracy",
                cv=inner_split,
                refit=True,
                n_jobs=1,
            )
            search.fit(matrix.reindex(columns=train_ids).T, y_train)
            models[model_name] = (
                search.best_estimator_,
                search.predict(matrix.reindex(columns=test_ids).T),
                search.predict_proba(matrix.reindex(columns=test_ids).T)[:, 1],
            )
            tuning_records.append({"model": model_name, "outer_fold": outer_fold, "best_c": search.best_params_["model__C"], "best_inner_balanced_accuracy": search.best_score_})
        forest = peptide_random_forest(
            minimum_prevalence,
            seed + int(outer_fold),
            int(config["random_forest"]["n_estimators"]),
            int(config["random_forest"]["min_samples_leaf"]),
        )
        forest.fit(peptide.reindex(columns=train_ids).T, y_train)
        models["peptide_random_forest"] = (
            forest,
            forest.predict(peptide.reindex(columns=test_ids).T),
            forest.predict_proba(peptide.reindex(columns=test_ids).T)[:, 1],
        )
        tuning_records.append({"model": "batch_logistic", "outer_fold": outer_fold, "best_c": batch_search.best_params_["model__C"], "best_inner_balanced_accuracy": batch_search.best_score_})
        for model_name, (fitted_model, predictions, probabilities) in models.items():
            metrics = binary_metrics(y_test, np.asarray(predictions), np.asarray(probabilities))
            fold_metrics.append({"model": model_name, "outer_fold": outer_fold, "test_samples": len(test_ids), **metrics})
            all_predictions.extend(prediction_rows(model_name, int(outer_fold), test_ids, test_labels, np.asarray(predictions), np.asarray(probabilities), positive_label))
            joblib.dump(fitted_model, checkpoint_dir / f"{model_name}_fold_{outer_fold}.joblib")
    metrics_frame = pd.DataFrame(fold_metrics)
    predictions_frame = pd.DataFrame(all_predictions)
    metric_columns = ["balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc", "sensitivity", "specificity", "accuracy"]
    summary = metrics_frame.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index().sort_values("balanced_accuracy_mean", ascending=False)
    metrics_frame.to_csv(table_dir / "stage_06_fold_metrics.csv", index=False)
    summary.to_csv(table_dir / "stage_06_model_summary.csv", index=False)
    predictions_frame.to_csv(table_dir / "stage_06_predictions.csv", index=False)
    pd.DataFrame(tuning_records).to_csv(table_dir / "stage_06_tuning.csv", index=False)
    comparison_plot(metrics_frame, figure_dir / "stage_06_baseline_comparison.png")
    best = summary.iloc[0]
    (report_dir / "stage_06_conventional_baselines.md").write_text("\n".join([
        "# Stage 6 — Conventional baseline models", "",
        "## Evaluation design", "",
        "- Five immutable outer folds grouped by raw file.",
        "- Three saved grouped inner folds tune logistic-regression regularization.",
        "- Feature prevalence filtering, log transformation, median imputation, and scaling are fitted inside training folds only.",
        f"- Positive class: `{positive_label}`; class weighting is enabled.", "",
        "## Models", "",
        "- Majority-class predictor.",
        "- Batch-only logistic diagnostic.",
        "- Protein fixed-vector logistic regression.",
        "- Peptide fixed-vector logistic regression.",
        "- Peptide random forest as the nonlinear baseline.", "",
        "## Primary result", "",
        f"- Best mean grouped balanced accuracy: `{MODEL_LABELS[str(best['model'])]}` = {best['balanced_accuracy_mean']:.3f} ± {best['balanced_accuracy_std']:.3f}.",
        "- Fold-level predictions and all secondary metrics are saved for later paired comparisons.", "",
        "## Interpretation", "",
        "- These are proof-of-concept results for the current 215-sample local corpus.",
        "- Performance must be compared against MIL using exactly the same outer folds.",
        "- Batch-holdout robustness remains a secondary evaluation and is not included in this primary summary.", "",
    ]), encoding="utf-8")
    run_metadata = {
        "positive_label": positive_label,
        "samples": int(metadata.shape[0]),
        "outer_folds": int(outer["outer_fold"].nunique()),
        "models": list(MODEL_LABELS),
    }
    (table_dir / "stage_06_run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    return {"samples": len(metadata), "models": len(MODEL_LABELS), "folds": int(outer["outer_fold"].nunique()), "best_model": str(best["model"]), "best_balanced_accuracy": float(best["balanced_accuracy_mean"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    logger = configure_logging(Path("outputs/logs/stage_06_baselines.log"))
    logger.info("Stage 6 completed: %s", run_stage_six(args.config))


if __name__ == "__main__":
    main()
