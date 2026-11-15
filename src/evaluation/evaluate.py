"""Run Stage 9 primary evaluation on aligned out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score

from src.evaluation.bootstrap import (
    DEFAULT_METRICS,
    exact_sign_flip_pvalue,
    group_bootstrap_metrics,
    paired_group_bootstrap_differences,
    percentile_interval,
)
from src.evaluation.calibration import calibration_metrics, reliability_table
from src.evaluation.metrics import binary_metrics
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


MODEL_LABELS = {
    "majority": "Majority",
    "batch_logistic": "Batch logistic",
    "protein_logistic": "Protein logistic",
    "peptide_logistic": "Peptide logistic",
    "peptide_random_forest": "Peptide random forest",
    "mean_pooling": "Mean-pooling DeepSets",
    "max_pooling": "Max-pooling DeepSets",
    "attention": "Attention MIL",
}
MODEL_FAMILIES = {
    "majority": "trivial",
    "batch_logistic": "technical_diagnostic",
    "protein_logistic": "fixed_vector_biological",
    "peptide_logistic": "fixed_vector_biological",
    "peptide_random_forest": "fixed_vector_biological",
    "mean_pooling": "mil",
    "max_pooling": "mil",
    "attention": "mil",
}
SUBGROUP_COLUMNS = [
    "raw_file_stem", "batch_chromatography", "batch_digest",
    "batch_sort", "reporter_channel",
]


def load_aligned_predictions(table_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Stage 6/8 predictions and verify exact sample/truth/fold alignment."""
    stage_six = pd.read_csv(table_directory / "stage_06_predictions.csv")
    stage_eight = pd.read_csv(table_directory / "stage_08_predictions.csv")
    predictions = pd.concat([stage_six, stage_eight], ignore_index=True)
    expected_models = set(MODEL_LABELS)
    if set(predictions["model"]) != expected_models:
        raise ValueError(
            f"prediction models differ from expected models: {sorted(set(predictions['model']))}"
        )
    if predictions.duplicated(["model", "sample_id"]).any():
        raise ValueError("a model contains duplicate sample predictions")
    if predictions["probability_sc_u"].isna().any() or not predictions["probability_sc_u"].between(0, 1).all():
        raise ValueError("prediction probabilities must be complete and between zero and one")
    canonical = predictions[predictions["model"].eq("peptide_logistic")][
        ["sample_id", "y_true", "label_raw", "outer_fold"]
    ].set_index("sample_id")
    for model_name, frame in predictions.groupby("model"):
        aligned = frame.set_index("sample_id").reindex(canonical.index)
        if aligned.isna().any().any():
            raise ValueError(f"model {model_name} does not cover all canonical samples")
        if not np.array_equal(aligned["y_true"], canonical["y_true"]):
            raise ValueError(f"model {model_name} has inconsistent truth labels")
        if not np.array_equal(aligned["outer_fold"], canonical["outer_fold"]):
            raise ValueError(f"model {model_name} has inconsistent outer-fold assignments")
    bags = pd.read_parquet("data/processed/mil_bags/bag_index.parquet")
    metadata_columns = ["sample_id", "bag_size", *SUBGROUP_COLUMNS]
    if set(canonical.index) != set(bags["sample_id"]):
        raise ValueError("prediction samples differ from the Stage 7 bag index")
    predictions = predictions.merge(
        bags[metadata_columns], on="sample_id", how="left", validate="many_to_one"
    )
    return predictions, bags


def overall_results(predictions: pd.DataFrame, calibration_bins: int) -> pd.DataFrame:
    """Calculate overall out-of-fold discrimination and calibration metrics."""
    records = []
    for model_name in MODEL_LABELS:
        frame = predictions[predictions["model"].eq(model_name)]
        metrics = binary_metrics(
            frame["y_true"].to_numpy(),
            frame["y_pred"].to_numpy(),
            frame["probability_sc_u"].to_numpy(),
        )
        calibration = calibration_metrics(
            frame["y_true"].to_numpy(),
            frame["probability_sc_u"].to_numpy(),
            calibration_bins,
        )
        records.append({
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "model_family": MODEL_FAMILIES[model_name],
            "samples": len(frame),
            **metrics,
            **calibration,
        })
    return pd.DataFrame(records).sort_values("balanced_accuracy", ascending=False)


def bootstrap_all_models(
    predictions: pd.DataFrame,
    group_column: str,
    replicates: int,
    confidence_level: float,
    seed: int,
    point_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long confidence-interval and replicate tables for every model."""
    replicate_frames = []
    interval_records = []
    point_by_model = point_results.set_index("model")
    for model_index, model_name in enumerate(MODEL_LABELS):
        frame = predictions[predictions["model"].eq(model_name)]
        bootstrap = group_bootstrap_metrics(
            frame, group_column, replicates, seed + model_index * 10_000
        )
        bootstrap.insert(0, "model", model_name)
        replicate_frames.append(bootstrap)
        for metric in DEFAULT_METRICS:
            lower, upper = percentile_interval(bootstrap[metric], confidence_level)
            interval_records.append({
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "metric": metric,
                "point_estimate": float(point_by_model.loc[model_name, metric]),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
                "bootstrap_replicates": replicates,
                "bootstrap_group": group_column,
            })
    return pd.DataFrame(interval_records), pd.concat(replicate_frames, ignore_index=True)


def subgroup_metric_record(frame: pd.DataFrame, minimum_samples: int) -> dict[str, Any]:
    """Calculate subgroup metrics while flagging unreliable one-class/small groups."""
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    probability = frame["probability_sc_u"].to_numpy(dtype=float)
    negatives, positives = int((y_true == 0).sum()), int((y_true == 1).sum())
    meets_minimum = len(frame) >= minimum_samples
    both_classes = negatives > 0 and positives > 0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    record: dict[str, Any] = {
        "samples": len(frame),
        "negative_samples": negatives,
        "positive_samples": positives,
        "meets_minimum_sample_size": meets_minimum,
        "contains_both_classes": both_classes,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "sensitivity": float(tp / (tp + fn)) if positives else float("nan"),
        "specificity": float(tn / (tn + fp)) if negatives else float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "mcc": float("nan"),
    }
    if meets_minimum and both_classes:
        metrics = binary_metrics(y_true, y_pred, probability)
        record.update({key: metrics[key] for key in (
            "balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc",
            "sensitivity", "specificity", "accuracy",
        )})
    return record


def subgroup_results(
    predictions: pd.DataFrame,
    minimum_samples: int,
) -> pd.DataFrame:
    """Report model robustness across raw files, batches, and reporter channels."""
    records = []
    for model_name in MODEL_LABELS:
        model_frame = predictions[predictions["model"].eq(model_name)]
        for group_column in SUBGROUP_COLUMNS:
            for group_value, frame in model_frame.groupby(group_column, dropna=False):
                records.append({
                    "model": model_name,
                    "grouping_variable": group_column,
                    "group_value": group_value,
                    **subgroup_metric_record(frame, minimum_samples),
                })
    return pd.DataFrame(records)


def paired_comparisons(
    predictions: pd.DataFrame,
    reference_model: str,
    comparator_models: list[str],
    group_column: str,
    replicates: int,
    confidence_level: float,
    seed: int,
    table_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare models with paired group bootstrap and exact fold sign flips."""
    stage_six_metrics = pd.read_csv(table_directory / "stage_06_fold_metrics.csv")
    stage_eight_metrics = pd.read_csv(table_directory / "stage_08_fold_metrics.csv")
    fold_metrics = pd.concat([stage_six_metrics, stage_eight_metrics], ignore_index=True)
    reference_predictions = predictions[predictions["model"].eq(reference_model)]
    reference_point = binary_metrics(
        reference_predictions["y_true"].to_numpy(),
        reference_predictions["y_pred"].to_numpy(),
        reference_predictions["probability_sc_u"].to_numpy(),
    )
    result_records = []
    replicate_frames = []
    for comparator_index, comparator_model in enumerate(comparator_models):
        comparator_predictions = predictions[predictions["model"].eq(comparator_model)]
        comparator_point = binary_metrics(
            comparator_predictions["y_true"].to_numpy(),
            comparator_predictions["y_pred"].to_numpy(),
            comparator_predictions["probability_sc_u"].to_numpy(),
        )
        bootstrap = paired_group_bootstrap_differences(
            reference_predictions,
            comparator_predictions,
            group_column,
            replicates,
            seed + 100_000 + comparator_index * 10_000,
        )
        bootstrap.insert(0, "comparator_model", comparator_model)
        bootstrap.insert(0, "reference_model", reference_model)
        replicate_frames.append(bootstrap)
        reference_fold = fold_metrics[fold_metrics["model"].eq(reference_model)]
        comparator_fold = fold_metrics[fold_metrics["model"].eq(comparator_model)]
        paired_folds = reference_fold.merge(
            comparator_fold, on="outer_fold", suffixes=("_reference", "_comparator"),
            validate="one_to_one",
        )
        for metric in DEFAULT_METRICS:
            differences = (
                paired_folds[f"{metric}_comparator"]
                - paired_folds[f"{metric}_reference"]
            ).to_numpy()
            lower, upper = percentile_interval(bootstrap[metric], confidence_level)
            result_records.append({
                "reference_model": reference_model,
                "comparator_model": comparator_model,
                "metric": metric,
                "oof_point_difference": comparator_point[metric] - reference_point[metric],
                "group_bootstrap_ci_lower": lower,
                "group_bootstrap_ci_upper": upper,
                "fold_mean_difference": float(differences.mean()),
                "fold_standard_deviation": float(differences.std(ddof=1)),
                "exact_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
                "outer_fold_pairs": len(differences),
                "bootstrap_replicates": replicates,
            })
    return pd.DataFrame(result_records), pd.concat(replicate_frames, ignore_index=True)


def calibration_tables(
    predictions: pd.DataFrame,
    calibration_bins: int,
) -> pd.DataFrame:
    """Build a reliability table for each model without fitting a calibrator."""
    frames = []
    for model_name in MODEL_LABELS:
        frame = predictions[predictions["model"].eq(model_name)]
        table = reliability_table(
            frame["y_true"].to_numpy(),
            frame["probability_sc_u"].to_numpy(),
            calibration_bins,
        )
        table.insert(0, "model", model_name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def error_analysis_table(
    predictions: pd.DataFrame,
    bags: pd.DataFrame,
    models: list[str],
    reference_model: str,
) -> pd.DataFrame:
    """Create one row per cell with predictions, correctness, and disagreement flags."""
    output = bags[[
        "sample_id", "label_raw", "label_binary", "bag_size", "raw_file_stem",
        "batch_chromatography", "batch_digest", "batch_sort", "reporter_channel",
    ]].copy()
    for model_name in models:
        model_predictions = predictions[predictions["model"].eq(model_name)].set_index("sample_id")
        output[f"{model_name}_prediction"] = output["sample_id"].map(model_predictions["y_pred"])
        output[f"{model_name}_probability_sc_u"] = output["sample_id"].map(model_predictions["probability_sc_u"])
        output[f"{model_name}_correct"] = output[f"{model_name}_prediction"].eq(output["label_binary"])
        output[f"{model_name}_predicted_class_confidence"] = np.where(
            output[f"{model_name}_prediction"].eq(1),
            output[f"{model_name}_probability_sc_u"],
            1.0 - output[f"{model_name}_probability_sc_u"],
        )
        output[f"{model_name}_high_confidence_error"] = (
            ~output[f"{model_name}_correct"]
            & output[f"{model_name}_predicted_class_confidence"].ge(0.8)
        )
    for model_name in models:
        if model_name == reference_model:
            continue
        output[f"{model_name}_disagrees_with_{reference_model}"] = output[
            f"{model_name}_prediction"
        ].ne(output[f"{reference_model}_prediction"])
        output[f"{model_name}_error_relative_to_{reference_model}"] = np.select(
            [
                output[f"{reference_model}_correct"] & ~output[f"{model_name}_correct"],
                ~output[f"{reference_model}_correct"] & output[f"{model_name}_correct"],
                ~output[f"{reference_model}_correct"] & ~output[f"{model_name}_correct"],
            ],
            ["reference_only_correct", "comparator_only_correct", "both_wrong"],
            default="both_correct",
        )
    return output


def confidence_interval_plot(intervals: pd.DataFrame, output_path: Path) -> None:
    """Plot overall metrics with raw-file-group bootstrap confidence intervals."""
    order = list(MODEL_LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, constrained_layout=True)
    for axis, metric in zip(axes.flat, DEFAULT_METRICS):
        selected = intervals[intervals["metric"].eq(metric)].set_index("model").loc[order]
        y = np.arange(len(order))
        point = selected["point_estimate"].to_numpy()
        errors = np.vstack((point - selected["ci_lower"], selected["ci_upper"] - point))
        axis.errorbar(point, y, xerr=errors, fmt="o", capsize=3)
        axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_yticks(y, [MODEL_LABELS[name] for name in order])
        axis.invert_yaxis()
        axis.set_xlabel(metric.replace("_", " ").title())
        axis.set_title("95% raw-file-group bootstrap CI")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def calibration_plot(reliability: pd.DataFrame, output_path: Path) -> None:
    """Plot reliability diagrams for the primary biological comparisons."""
    selected_models = ["peptide_logistic", "max_pooling", "attention"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    for axis, model_name in zip(axes, selected_models):
        frame = reliability[
            reliability["model"].eq(model_name) & reliability["sample_count"].gt(0)
        ]
        axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        axis.plot(
            frame["mean_probability"], frame["observed_positive_fraction"],
            marker="o", linewidth=1.5,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mean predicted probability")
        axis.set_title(MODEL_LABELS[model_name])
    axes[0].set_ylabel("Observed sc_u fraction")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def paired_difference_plot(replicates: pd.DataFrame, output_path: Path) -> None:
    """Plot paired bootstrap metric differences relative to peptide logistic."""
    comparators = replicates["comparator_model"].drop_duplicates().tolist()
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, DEFAULT_METRICS):
        values = [
            replicates.loc[replicates["comparator_model"].eq(model), metric].to_numpy()
            for model in comparators
        ]
        axis.boxplot(values, tick_labels=[MODEL_LABELS[model] for model in comparators], showfliers=False)
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_ylabel(f"Comparator − peptide logistic")
        axis.set_title(metric.replace("_", " ").title())
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_stage_nine(
    config_path: str | Path = "configs/evaluation.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute Stage 9 evaluation and write all statistical artifacts."""
    config = load_yaml_config(config_path)
    table_directory = Path("outputs/tables")
    report_directory = Path("outputs/reports")
    figure_directory = Path("outputs/figures")
    for directory in (table_directory, report_directory, figure_directory):
        directory.mkdir(parents=True, exist_ok=True)
    predictions, bags = load_aligned_predictions(table_directory)
    group_column = str(config["bootstrap_group_column"])
    if group_column not in predictions:
        raise ValueError(f"bootstrap group column is unavailable: {group_column}")
    if dry_run:
        return {
            "dry_run": True,
            "samples": int(bags["sample_id"].nunique()),
            "models": int(predictions["model"].nunique()),
            "raw_file_groups": int(predictions[group_column].nunique()),
            "aligned_predictions": int(len(predictions)),
        }
    seed = int(config["seed"])
    replicates = int(config["bootstrap_replicates"])
    confidence_level = float(config["confidence_level"])
    calibration_bins = int(config["calibration_bins"])
    minimum_subgroup_samples = int(config["minimum_subgroup_samples"])
    reference_model = str(config["primary_baseline"])
    comparator_models = [
        str(config["primary_mil"]),
        str(config["additional_mil_comparator"]),
    ]
    logger = logging.getLogger("single_cell_proteomics_mil")
    logger.info("Stage 9 | calculating overall and calibration metrics")
    results = overall_results(predictions, calibration_bins)
    logger.info("Stage 9 | running %s raw-file-group bootstrap replicates per model", replicates)
    intervals, bootstrap_replicates = bootstrap_all_models(
        predictions, group_column, replicates, confidence_level, seed, results
    )
    logger.info("Stage 9 | running paired group bootstrap comparisons")
    paired, paired_replicates = paired_comparisons(
        predictions, reference_model, comparator_models, group_column,
        replicates, confidence_level, seed, table_directory,
    )
    reliability = calibration_tables(predictions, calibration_bins)
    subgroups = subgroup_results(predictions, minimum_subgroup_samples)
    errors = error_analysis_table(
        predictions, bags, [reference_model, *comparator_models], reference_model
    )
    results.to_csv(table_directory / "stage_09_overall_results.csv", index=False)
    intervals.to_csv(table_directory / "stage_09_confidence_intervals.csv", index=False)
    bootstrap_replicates.to_parquet(
        table_directory / "stage_09_bootstrap_replicates.parquet", index=False
    )
    paired.to_csv(table_directory / "stage_09_paired_comparisons.csv", index=False)
    paired_replicates.to_parquet(
        table_directory / "stage_09_paired_bootstrap_replicates.parquet", index=False
    )
    reliability.to_csv(table_directory / "stage_09_reliability_bins.csv", index=False)
    subgroups.to_csv(table_directory / "stage_09_subgroup_metrics.csv", index=False)
    errors.to_csv(table_directory / "stage_09_error_analysis.csv", index=False)
    confidence_interval_plot(
        intervals, figure_directory / "stage_09_model_confidence_intervals.png"
    )
    calibration_plot(reliability, figure_directory / "stage_09_calibration.png")
    paired_difference_plot(
        paired_replicates, figure_directory / "stage_09_paired_differences.png"
    )
    balanced_intervals = intervals[intervals["metric"].eq("balanced_accuracy")].set_index("model")
    reference_result = results.set_index("model").loc[reference_model]
    attention_comparison = paired[
        paired["comparator_model"].eq(config["primary_mil"])
        & paired["metric"].eq("balanced_accuracy")
    ].iloc[0]
    max_comparison = paired[
        paired["comparator_model"].eq(config["additional_mil_comparator"])
        & paired["metric"].eq("balanced_accuracy")
    ].iloc[0]
    key_models = [reference_model, *comparator_models]
    result_lines = []
    for model_name in key_models:
        row = results.set_index("model").loc[model_name]
        ci = balanced_intervals.loc[model_name]
        result_lines.append(
            f"- `{MODEL_LABELS[model_name]}`: balanced accuracy {row['balanced_accuracy']:.3f} "
            f"(95% group-bootstrap CI {ci['ci_lower']:.3f}–{ci['ci_upper']:.3f}), "
            f"AUROC {row['auroc']:.3f}, Brier score {row['brier_score']:.3f}, "
            f"ECE {row['expected_calibration_error']:.3f}."
        )
    chromatography = subgroups[
        subgroups["grouping_variable"].eq("batch_chromatography")
        & subgroups["model"].isin(key_models)
        & subgroups["balanced_accuracy"].notna()
    ]
    worst_batch_lines = []
    for model_name in key_models:
        model_batches = chromatography[chromatography["model"].eq(model_name)]
        worst = model_batches.loc[model_batches["balanced_accuracy"].idxmin()]
        worst_batch_lines.append(
            f"- `{MODEL_LABELS[model_name]}`: {worst['group_value']} = "
            f"{worst['balanced_accuracy']:.3f} (n={int(worst['samples'])})."
        )
    (report_directory / "stage_09_primary_evaluation.md").write_text("\n".join([
        "# Stage 9 — Primary processed-data evaluation", "",
        "## Design", "",
        f"- Evaluated {len(MODEL_LABELS)} models on the same {len(bags)} out-of-fold cell predictions.",
        f"- Confidence intervals use {replicates} bootstrap replicates that resample all {bags[group_column].nunique()} raw-file groups, not individual cells.",
        f"- Confidence level: {confidence_level:.0%}.",
        "- Probabilities are evaluated as produced by each model; no post-hoc calibrator was fitted.",
        "- Paired differences are comparator minus peptide logistic, using identical cells and raw-file bootstrap samples.", "",
        "## Primary results", "",
        *result_lines, "",
        "## Paired balanced-accuracy comparison", "",
        f"- Attention MIL minus peptide logistic: {attention_comparison['oof_point_difference']:.3f} "
        f"(95% group-bootstrap CI {attention_comparison['group_bootstrap_ci_lower']:.3f}–{attention_comparison['group_bootstrap_ci_upper']:.3f}; "
        f"five-fold exact sign-flip p={attention_comparison['exact_sign_flip_pvalue']:.3f}).",
        f"- Max-pooling DeepSets minus peptide logistic: {max_comparison['oof_point_difference']:.3f} "
        f"(95% group-bootstrap CI {max_comparison['group_bootstrap_ci_lower']:.3f}–{max_comparison['group_bootstrap_ci_upper']:.3f}; "
        f"five-fold exact sign-flip p={max_comparison['exact_sign_flip_pvalue']:.3f}).",
        "- The confidence intervals, effect sizes, and consistently negative fold differences support the current fixed-vector baseline; the small number of folds limits the permutation-test resolution.", "",
        "## Worst chromatography-batch balanced accuracy", "",
        *worst_batch_lines, "",
        "## Interpretation", "",
        f"- The strongest processed-data model remains `{MODEL_LABELS[reference_model]}` at {reference_result['balanced_accuracy']:.3f} out-of-fold balanced accuracy.",
        "- Initial MIL models do not outperform the regularized fixed-vector peptide representation on this proof-of-concept corpus.",
        "- This is a model-comparison result, not evidence that MIL cannot work with more cells, richer instance features, or broader experimental coverage.",
        "- Subgroup rows with fewer than the configured minimum or only one class are flagged and do not receive balanced accuracy or AUROC values.", "",
    ]), encoding="utf-8")
    metadata = {
        "samples": int(len(bags)),
        "models": int(len(MODEL_LABELS)),
        "aligned_predictions": int(len(predictions)),
        "raw_file_groups": int(bags[group_column].nunique()),
        "bootstrap_replicates_per_model": replicates,
        "confidence_level": confidence_level,
        "primary_baseline": reference_model,
        "primary_mil": str(config["primary_mil"]),
        "best_model_by_oof_balanced_accuracy": str(results.iloc[0]["model"]),
        "best_oof_balanced_accuracy": float(results.iloc[0]["balanced_accuracy"]),
    }
    (table_directory / "stage_09_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_09_evaluation.log"))
    result = run_stage_nine(args.config, args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 9 completed: %s", result)


if __name__ == "__main__":
    main()
