"""Execute Stage 17 group-aware learning curves and data-sufficiency analysis."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.build_splits import make_inner_assignments, make_outer_assignments
from src.evaluation.bootstrap import group_bootstrap_metrics, paired_group_bootstrap_differences, percentile_interval
from src.evaluation.metrics import binary_metrics
from src.training.spectrum_datasets import SpectrumArrayStore, make_psm_vocabulary
from src.training.train_expanded_models import (
    MODEL_FAMILIES as STAGE15_FAMILIES,
    MODEL_LABELS as STAGE15_LABELS,
    PROGRESS_NAMES as STAGE15_PROGRESS_NAMES,
    averaged_predictions,
    load_progress as load_stage15_progress,
    make_peptide_bags,
    train_conventional,
    train_peptide_mil,
    train_spectrum_mil,
    validate_predictions,
)
from src.training.train_mil import resolve_device
from src.training.train_multimodal_mil import (
    MODEL_LABELS as STAGE16_LABELS,
    PROGRESS_NAMES as STAGE16_PROGRESS_NAMES,
    average_predictions as average_hybrid_predictions,
    load_progress as load_stage16_progress,
    run_late_fusion,
    run_trainable_hybrids,
)
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


HYBRID_MODELS = ("late_fusion", "joint_hybrid")
MODEL_LABELS = STAGE15_LABELS | {name: STAGE16_LABELS[name] for name in HYBRID_MODELS}
MODEL_FAMILIES = STAGE15_FAMILIES | {"late_fusion": "multimodal", "joint_hybrid": "multimodal"}
METRICS = ("balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc")


def _group_table(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest.groupby("raw_file_stem", as_index=False).agg(
        cells=("sample_id", "size"),
        positive_cells=("label_raw", lambda values: int(values.eq("sc_u").sum())),
        negative_cells=("label_raw", lambda values: int(values.eq("sc_m0").sum())),
        batch_chromatography=("batch_chromatography", "first"),
    )


def select_group_complete_cohort(
    manifest: pd.DataFrame, target_cells: int, seed: int, trials: int = 20_000,
    forbidden_signatures: set[tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Choose a diverse, whole-raw-file cohort closest to the requested cell count."""
    groups = _group_table(manifest)
    if target_cells >= len(manifest):
        return manifest.copy()
    rng = np.random.default_rng(seed)
    batches = sorted(groups["batch_chromatography"].astype(str).unique())
    forbidden = forbidden_signatures or set()
    best: tuple[tuple[Any, ...], tuple[str, ...]] | None = None
    for _ in range(trials):
        mandatory = []
        for batch in batches:
            choices = groups.index[groups["batch_chromatography"].astype(str).eq(batch)].to_numpy()
            mandatory.append(int(rng.choice(choices)))
        mandatory = sorted(set(mandatory))
        remaining = np.setdiff1d(groups.index.to_numpy(), np.asarray(mandatory), assume_unique=False)
        order = np.concatenate((np.asarray(mandatory), rng.permutation(remaining)))
        cumulative = groups.loc[order, "cells"].to_numpy(int).cumsum()
        candidates = np.arange(len(mandatory) - 1, len(order))
        end = int(candidates[np.argmin(np.abs(cumulative[candidates] - target_cells))])
        selected = groups.loc[order[: end + 1]]
        signature = tuple(sorted(selected["raw_file_stem"].astype(str)))
        if signature in forbidden:
            continue
        total = int(selected["cells"].sum())
        positive_rate = float(selected["positive_cells"].sum() / total)
        full_rate = float(manifest["label_raw"].eq("sc_u").mean())
        score = (abs(total - target_cells), -int(selected["batch_chromatography"].nunique()), abs(positive_rate - full_rate), rng.random())
        if best is None or score < best[0]:
            best = (score, signature)
            if score[0] == 0 and score[1] == -len(batches) and score[2] < 0.005:
                break
    if best is None:
        raise RuntimeError(f"could not construct a unique cohort near {target_cells} cells")
    cohort = manifest[manifest["raw_file_stem"].astype(str).isin(set(best[1]))].copy()
    if cohort["label_raw"].nunique() != 2:
        raise AssertionError("cohort lacks both labels")
    if cohort["batch_chromatography"].nunique() != manifest["batch_chromatography"].nunique():
        raise AssertionError("cohort does not preserve maximum batch coverage")
    return cohort


def build_cohorts(manifest: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments, summaries = [], []
    for target in (int(value) for value in config["target_cells"]):
        repeats = int(config["full_corpus_repeats"] if target >= len(manifest) else config["subsampling_repeats"])
        signatures: set[tuple[str, ...]] = set()
        for repeat in range(repeats):
            cohort = select_group_complete_cohort(
                manifest, target, int(config["seed"]) + target * 100 + repeat,
                int(config["cohort_search_trials"]), signatures,
            )
            signatures.add(tuple(sorted(cohort["raw_file_stem"].astype(str).unique())))
            cohort_id = f"n{target}_r{repeat}"
            selected = cohort[["sample_id", "label_raw", "raw_file_stem", "batch_chromatography"]].copy()
            selected.insert(0, "cohort_id", cohort_id)
            selected.insert(1, "target_cells", target)
            selected.insert(2, "repeat", repeat)
            assignments.append(selected)
            counts = cohort["label_raw"].value_counts()
            summaries.append({
                "cohort_id": cohort_id, "target_cells": target, "repeat": repeat,
                "actual_cells": len(cohort), "raw_file_groups": cohort["raw_file_stem"].nunique(),
                "chromatography_batches": cohort["batch_chromatography"].nunique(),
                "sc_m0_cells": int(counts.get("sc_m0", 0)), "sc_u_cells": int(counts.get("sc_u", 0)),
                "positive_fraction": float(counts.get("sc_u", 0) / len(cohort)),
            })
    return pd.concat(assignments, ignore_index=True), pd.DataFrame(summaries)


def build_splits(manifest: pd.DataFrame, cohort_assignments: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_frames, inner_frames = [], []
    columns = ["sample_id", "label_raw", "raw_file_stem"]
    for cohort_id, assigned in cohort_assignments.groupby("cohort_id", sort=False):
        cohort = manifest[manifest["sample_id"].isin(assigned["sample_id"])][columns].copy()
        target, repeat = int(assigned["target_cells"].iloc[0]), int(assigned["repeat"].iloc[0])
        base_seed = int(config["seed"]) + target * 10 + repeat
        last_error: Exception | None = None
        for offset in range(100):
            try:
                outer = make_outer_assignments(cohort, "raw_file_stem", int(config["outer_folds"]), base_seed + offset)
                inner = make_inner_assignments(outer, "raw_file_stem", int(config["inner_folds"]), base_seed + offset)
                break
            except (AssertionError, ValueError) as error:
                last_error = error
        else:
            raise RuntimeError(f"could not construct valid grouped folds for {cohort_id}: {last_error}")
        for frame in (outer, inner):
            frame.insert(0, "cohort_id", cohort_id); frame.insert(1, "target_cells", target); frame.insert(2, "repeat", repeat)
        outer_frames.append(outer); inner_frames.append(inner)
    return pd.concat(outer_frames, ignore_index=True), pd.concat(inner_frames, ignore_index=True)


def validate_design(manifest: pd.DataFrame, cohorts: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame) -> None:
    if not cohorts["sample_id"].isin(manifest["sample_id"]).all():
        raise AssertionError("cohort contains an unknown cell")
    for cohort_id, assigned in cohorts.groupby("cohort_id"):
        ids, expected_groups = set(assigned["sample_id"]), set(assigned["raw_file_stem"])
        cohort_outer = outer[outer["cohort_id"].eq(cohort_id)]
        counts = cohort_outer[cohort_outer["partition"].eq("test")]["sample_id"].value_counts()
        if set(counts.index) != ids or not counts.eq(1).all():
            raise AssertionError(f"{cohort_id}: each cell must appear once in outer test data")
        for _, fold in cohort_outer.groupby("outer_fold"):
            if fold.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
                raise AssertionError(f"{cohort_id}: raw-file leakage in outer folds")
            if set(fold["raw_file_stem"]) != expected_groups:
                raise AssertionError(f"{cohort_id}: an outer fold partially drops a raw group")
        for _, fold in inner[inner["cohort_id"].eq(cohort_id)].groupby(["outer_fold", "inner_fold"]):
            if fold.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
                raise AssertionError(f"{cohort_id}: raw-file leakage in inner folds")


def _add_context(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    for key, value in reversed(list(context.items())):
        output.insert(0, key, value)
    return output


def _load_completed(progress_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(progress_dir / f"{name}.parquet") for name in ("predictions", "fold_metrics", "runtime", "tuning", "history")}


def _reuse_full_corpus(context: dict[str, Any]) -> dict[str, pd.DataFrame]:
    stage16_filter = "model in @HYBRID_MODELS"
    results = {
        "predictions": pd.concat([pd.read_csv("outputs/tables/stage_15_predictions.csv"), pd.read_csv("outputs/tables/stage_16_predictions.csv").query(stage16_filter).assign(model_family="multimodal")], ignore_index=True, sort=False),
        "fold_metrics": pd.concat([pd.read_csv("outputs/tables/stage_15_fold_seed_metrics.csv"), pd.read_csv("outputs/tables/stage_16_fold_seed_metrics.csv").query(stage16_filter)], ignore_index=True, sort=False),
        "runtime": pd.concat([pd.read_csv("outputs/tables/stage_15_runtime_memory.csv"), pd.read_csv("outputs/tables/stage_16_runtime_memory.csv").query(stage16_filter)], ignore_index=True, sort=False),
        "tuning": pd.concat([pd.read_csv("outputs/tables/stage_15_tuning.csv"), pd.read_csv("outputs/tables/stage_16_fusion_tuning.csv").query(stage16_filter)], ignore_index=True, sort=False),
        "history": pd.concat([pd.read_csv("outputs/tables/stage_15_training_history.csv"), pd.read_csv("outputs/tables/stage_16_training_history.csv").query("model == 'joint_hybrid'")], ignore_index=True, sort=False),
    }
    return {name: _add_context(frame, context) for name, frame in results.items()}


def run_cohort(
    cohort_manifest: pd.DataFrame, peptide: pd.DataFrame, protein: pd.DataFrame,
    spectrum_bags: pd.DataFrame, spectrum_metadata: pd.DataFrame,
    peptide_vocabulary: pd.DataFrame, peptide_instances: pd.DataFrame, peptide_bags: pd.DataFrame,
    outer: pd.DataFrame, inner: pd.DataFrame, config: dict[str, Any], stage15_base: dict[str, Any],
    stage16_base: dict[str, Any], context: dict[str, Any], progress_dir: Path, checkpoint_root: Path,
) -> dict[str, pd.DataFrame]:
    if (progress_dir / "complete.json").exists():
        return _load_completed(progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    checkpoint15, checkpoint16 = checkpoint_root / "stage_15", checkpoint_root / "stage_16"
    checkpoint15.mkdir(parents=True, exist_ok=True); checkpoint16.mkdir(parents=True, exist_ok=True)
    config15 = copy.deepcopy(stage15_base)
    config15["data"]["peptide_bag_directory"] = str(config["data"]["peptide_bag_directory"])
    config15["runtime"]["resume"] = bool(config["resume"])
    device = resolve_device(str(config15["runtime"]["device"]))
    stage15_dir = progress_dir / "stage_15"; stage15_dir.mkdir(exist_ok=True)
    progress15 = load_stage15_progress(stage15_dir, bool(config["resume"]))
    train_conventional(config15, cohort_manifest, peptide, protein, outer, inner, checkpoint15, stage15_dir, progress15)
    train_peptide_mil(config15, peptide_vocabulary, peptide_instances, peptide_bags, outer, inner, checkpoint15, stage15_dir, progress15, device)
    directory = Path(config["data"]["directory"])
    spectra_path, reporter_path = directory / "spectra.h5", directory / "reporter_intensities.h5"
    train_spectrum_mil(config15, spectrum_bags, spectrum_metadata, spectra_path, reporter_path, outer, inner, checkpoint15, stage15_dir, progress15, device)
    predictions15, tuning15 = pd.DataFrame(progress15["predictions"]), pd.DataFrame(progress15["tuning"])
    validate_predictions(predictions15, set(cohort_manifest["sample_id"]), int(config15["seed"]), [int(value) for value in config15["neural_seeds"]])
    config16 = copy.deepcopy(stage16_base)
    config16["data"]["stage_15_checkpoints"] = str(checkpoint15)
    config16["training"]["variants"] = ["joint_hybrid"]
    config16["runtime"]["resume"] = bool(config["resume"])
    stage16_dir = progress_dir / "stage_16"; stage16_dir.mkdir(exist_ok=True)
    progress16 = load_stage16_progress(stage16_dir, bool(config["resume"]))
    peptide_to_id, psm_vocabulary = make_psm_vocabulary(spectrum_metadata)
    vocabulary_size = int(psm_vocabulary["peptide_id"].max()) + 1 if len(psm_vocabulary) else 1
    array_store = SpectrumArrayStore.from_hdf5(spectra_path, reporter_path, int(config16["spectrum"]["spectrum_downsample_factor"]))
    run_late_fusion(config16, cohort_manifest, protein, spectrum_bags, spectrum_metadata, outer, spectra_path, reporter_path, array_store, peptide_to_id, vocabulary_size, predictions15, tuning15, checkpoint16, stage16_dir, progress16, device)
    run_trainable_hybrids(config16, cohort_manifest, protein, spectrum_bags, spectrum_metadata, outer, inner, spectra_path, reporter_path, array_store, peptide_to_id, vocabulary_size, tuning15, checkpoint16, stage16_dir, progress16, device)
    predictions16 = pd.DataFrame(progress16["predictions"])
    expected = set(cohort_manifest["sample_id"])
    for (model, seed), frame in predictions16.groupby(["model", "seed"]):
        if set(frame["sample_id"]) != expected or frame["sample_id"].duplicated().any():
            raise AssertionError(f"{model}/{seed} lacks complete OOF coverage")
    predictions16["model_family"] = "multimodal"
    results = {
        "predictions": pd.concat([predictions15, predictions16], ignore_index=True, sort=False),
        "fold_metrics": pd.concat([pd.DataFrame(progress15["fold_metrics"]), pd.DataFrame(progress16["fold_metrics"])], ignore_index=True, sort=False),
        "runtime": pd.concat([pd.DataFrame(progress15["runtime"]), pd.DataFrame(progress16["runtime"])], ignore_index=True, sort=False),
        "tuning": pd.concat([tuning15, pd.DataFrame(progress16["tuning"])], ignore_index=True, sort=False),
        "history": pd.concat([pd.DataFrame(progress15["history"]), pd.DataFrame(progress16["history"])], ignore_index=True, sort=False),
    }
    results = {name: _add_context(frame, context) for name, frame in results.items()}
    for name, frame in results.items():
        frame.to_parquet(progress_dir / f"{name}.parquet", index=False)
    (progress_dir / "complete.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    if not bool(config["retain_checkpoints"]):
        shutil.rmtree(checkpoint_root)
        for name in STAGE15_PROGRESS_NAMES:
            (stage15_dir / f"stage_15_{name}.partial.csv").unlink(missing_ok=True)
        for name in STAGE16_PROGRESS_NAMES:
            (stage16_dir / f"stage_16_{name}.partial.csv").unlink(missing_ok=True)
    return results


def averaged_all_models(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    first = averaged_predictions(predictions[predictions["model"].isin(STAGE15_LABELS)], manifest)
    second = average_hybrid_predictions(predictions[predictions["model"].isin(HYBRID_MODELS)], manifest)
    second["model_family"] = "multimodal"
    return pd.concat([first, second], ignore_index=True, sort=False)


def evaluate_learning_curves(predictions: pd.DataFrame, manifest: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows, interval_rows, paired_rows, bootstrap_frames, averaged_frames = [], [], [], [], []
    for cohort_index, (cohort_id, raw) in enumerate(predictions.groupby("cohort_id", sort=False)):
        cohort_manifest = manifest[manifest["sample_id"].isin(raw["sample_id"].unique())]
        averaged = averaged_all_models(raw, cohort_manifest)
        context = raw[["target_cells", "actual_cells", "repeat", "cohort_id"]].iloc[0].to_dict()
        averaged = _add_context(averaged, context); averaged_frames.append(averaged)
        points: dict[str, dict[str, float]] = {}
        for model_index, (model, frame) in enumerate(averaged.groupby("model", sort=True)):
            metrics = binary_metrics(frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int), frame["probability_sc_u"].to_numpy(float)); points[model] = metrics
            repeat_rows.append({**context, "model": model, "model_label": MODEL_LABELS[model], "model_family": MODEL_FAMILIES[model], "raw_file_groups": frame["raw_file_stem"].nunique(), **metrics})
            bootstrap = group_bootstrap_metrics(frame, "raw_file_stem", int(config["bootstrap_replicates"]), int(config["seed"]) + cohort_index * 1000 + model_index)
            bootstrap_frames.append(_add_context(bootstrap, context | {"model": model}))
            for metric in METRICS:
                lower, upper = percentile_interval(bootstrap[metric], float(config["confidence_level"]))
                interval_rows.append({**context, "model": model, "metric": metric, "point_estimate": metrics[metric], "ci_lower": lower, "ci_upper": upper, "bootstrap_replicates": int(config["bootstrap_replicates"])})
        reference_name = str(config["reference_model"]); reference = averaged[averaged["model"].eq(reference_name)]
        for model_index, model in enumerate(name for name in MODEL_LABELS if name != reference_name):
            comparator = averaged[averaged["model"].eq(model)]
            paired = paired_group_bootstrap_differences(reference, comparator, "raw_file_stem", int(config["bootstrap_replicates"]), int(config["seed"]) + 500_000 + cohort_index * 1000 + model_index)
            for metric in METRICS:
                lower, upper = percentile_interval(paired[metric], float(config["confidence_level"]))
                paired_rows.append({**context, "reference_model": reference_name, "comparator_model": model, "metric": metric, "point_difference_comparator_minus_reference": points[model][metric] - points[reference_name][metric], "ci_lower": lower, "ci_upper": upper})
    return pd.DataFrame(repeat_rows), pd.DataFrame(interval_rows), pd.DataFrame(paired_rows), pd.concat(bootstrap_frames, ignore_index=True), pd.concat(averaged_frames, ignore_index=True)


def summarize_learning_curves(repeat_metrics: pd.DataFrame, intervals: pd.DataFrame, cohort_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), frame in repeat_metrics.groupby(["target_cells", "model"]):
        summary: dict[str, Any] = {"estimate_type": "observed", "target_cells": int(target), "actual_cells_mean": float(frame["actual_cells"].mean()), "actual_cells_min": int(frame["actual_cells"].min()), "actual_cells_max": int(frame["actual_cells"].max()), "raw_file_groups_mean": float(frame["raw_file_groups"].mean()), "repeats": len(frame), "model": model, "model_label": MODEL_LABELS[model], "model_family": MODEL_FAMILIES[model]}
        for metric in METRICS:
            summary[f"{metric}_mean"] = float(frame[metric].mean())
            summary[f"{metric}_std_across_repeats"] = float(frame[metric].std(ddof=1)) if len(frame) > 1 else np.nan
            selected = intervals[intervals["target_cells"].eq(target) & intervals["model"].eq(model) & intervals["metric"].eq(metric)]
            summary[f"{metric}_ci_lower"] = float(selected["ci_lower"].mean()); summary[f"{metric}_ci_upper"] = float(selected["ci_upper"].mean())
        rows.append(summary)
    observed = pd.DataFrame(rows); extrapolated = []
    for model, frame in observed.groupby("model"):
        x = 1.0 / np.sqrt(frame["actual_cells_mean"].to_numpy(float))
        for future in (1200, 1500, 2000):
            record: dict[str, Any] = {"estimate_type": "descriptive_extrapolation", "target_cells": future, "actual_cells_mean": future, "actual_cells_min": np.nan, "actual_cells_max": np.nan, "raw_file_groups_mean": np.nan, "repeats": 0, "model": model, "model_label": MODEL_LABELS[model], "model_family": MODEL_FAMILIES[model]}
            for metric in METRICS:
                y = frame[f"{metric}_mean"].to_numpy(float)
                intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
                record[f"{metric}_mean"] = float(np.clip(intercept + slope / np.sqrt(future), -1 if metric == "mcc" else 0, 1))
                record[f"{metric}_std_across_repeats"] = np.nan; record[f"{metric}_ci_lower"] = np.nan; record[f"{metric}_ci_upper"] = np.nan
            extrapolated.append(record)
    return pd.concat([observed, pd.DataFrame(extrapolated)], ignore_index=True)


def add_fold_counts(fold_metrics: pd.DataFrame, outer: pd.DataFrame) -> pd.DataFrame:
    counts = outer.groupby(["cohort_id", "outer_fold", "partition"]).size().unstack(fill_value=0).reset_index().rename(columns={"train": "outer_train_cells", "test": "outer_test_cells"})
    groups = outer.groupby(["cohort_id", "outer_fold", "partition"])["raw_file_stem"].nunique().unstack(fill_value=0).reset_index().rename(columns={"train": "outer_train_raw_file_groups", "test": "outer_test_raw_file_groups"})
    return fold_metrics.merge(counts, on=["cohort_id", "outer_fold"], how="left").merge(groups, on=["cohort_id", "outer_fold"], how="left")


def compute_scaling_table(runtime: pd.DataFrame) -> pd.DataFrame:
    table = runtime.groupby(["target_cells", "actual_cells", "repeat", "model"], as_index=False)["runtime_seconds"].sum(); table["runtime_minutes"] = table["runtime_seconds"] / 60
    slopes = {model: float(np.polyfit(np.log(frame["actual_cells"]), np.log(frame["runtime_seconds"].clip(lower=1e-3)), 1)[0]) for model, frame in table.groupby("model") if frame["actual_cells"].nunique() > 1}
    table["empirical_runtime_power_exponent"] = table["model"].map(slopes)
    return table


def make_figures(learning: pd.DataFrame, compute: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True); observed = learning[learning["estimate_type"].eq("observed")]
    for x_column, filename, xlabel in (("actual_cells_mean", "stage_17_performance_vs_cells.png", "Cells in admitted corpus"), ("raw_file_groups_mean", "stage_17_performance_vs_raw_files.png", "Raw-file groups in admitted corpus")):
        figure, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
        for model, frame in observed.groupby("model"):
            frame = frame.sort_values(x_column); axis.plot(frame[x_column], frame["balanced_accuracy_mean"], marker="o", linewidth=1.4, label=MODEL_LABELS[model])
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1); axis.set(xlabel=xlabel, ylabel="Out-of-fold balanced accuracy", ylim=(0.4, 1.0), title="Stage 17 group-aware learning curves"); axis.legend(fontsize=7, ncol=2)
        figure.savefig(output / filename, dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for model, frame in compute.groupby("model"):
        mean = frame.groupby("actual_cells", as_index=False)["runtime_minutes"].mean(); axis.plot(mean["actual_cells"], mean["runtime_minutes"], marker="o", label=MODEL_LABELS.get(model, model))
    axis.set(xlabel="Cells in admitted corpus", ylabel="Mean measured runtime (minutes)", title="Stage 17 compute scaling"); axis.legend(fontsize=7, ncol=2)
    figure.savefig(output / "stage_17_compute_scaling.png", dpi=180); plt.close(figure)


def render_report(summary: dict[str, Any], learning: pd.DataFrame) -> str:
    observed = learning[learning["estimate_type"].eq("observed")]; maximum = observed[observed["target_cells"].eq(observed["target_cells"].max())].sort_values("balanced_accuracy_mean", ascending=False); best = maximum.iloc[0]
    lines = ["# Stage 17 — Group-aware learning curves and data sufficiency", "", "## Design", "", f"- Evaluated {summary['models']} models at target corpus sizes {', '.join(map(str, summary['targets']))}.", f"- Smaller targets use {summary['subsampling_repeats']} deterministic whole-raw-file subsamples; the complete {summary['maximum_cells']:,}-cell corpus is reused once because it has no distinct group-complete subsample.", "- Every cohort preserves all available chromatography batches, both labels, and complete raw-file groups.", "- Every model at a given size/repeat uses identical cells and grouped outer/inner folds.", "- Preprocessing, hyperparameter selection, and neural epoch selection are repeated inside each size-specific training partition.", f"- Confidence intervals use {summary['bootstrap_replicates']} raw-file-group bootstrap replicates per model and cohort.", "", "## Findings", "", f"- Best model at {summary['maximum_cells']:,} cells: `{best['model_label']}` with balanced accuracy {best['balanced_accuracy_mean']:.3f}."]
    for model in ("peptide_logistic", "protein_logistic", "late_fusion", "joint_hybrid", "spectrum_psm"):
        frame = observed[observed["model"].eq(model)].sort_values("actual_cells_mean"); first, last = frame.iloc[0], frame.iloc[-1]
        lines.append(f"- `{MODEL_LABELS[model]}`: {first['balanced_accuracy_mean']:.3f} at about {first['actual_cells_mean']:.0f} cells and {last['balanced_accuracy_mean']:.3f} at {last['actual_cells_mean']:.0f} cells.")
    lines += ["", "## Interpretation", "", "- Observed rows are empirical grouped out-of-fold results. Rows marked `descriptive_extrapolation` are simple fits against 1/sqrt(cells); they are not validated performance claims.", "- Variation across smaller-corpus repeats measures sensitivity to which independent raw-file groups were admitted.", "- Additional cells are most likely to help models whose observed curves continue to rise; flat or unstable curves suggest that batch diversity, label balance, or representation quality may matter more than cell count alone.", "", "## Outputs and integrity", "", f"- Saved {summary['cohorts']} cohort definitions, {summary['fold_metric_rows']:,} per-fold/per-seed metric rows, and {summary['prediction_rows']:,} seed-specific OOF predictions.", "- Automated checks found no raw-file group crossing an outer or inner partition and no partial-cell or duplicated-cell admission.", f"- Total measured/reused training runtime represented in the tables: {summary['runtime_hours']:.2f} hours.", ""]
    return "\n".join(lines)


def run_stage_seventeen(config_path: str | Path = "configs/stage_17.yaml", dry_run: bool = False) -> dict[str, Any]:
    config = load_yaml_config(config_path); manifest = pd.read_parquet(config["data"]["manifest"])
    cohorts, cohort_summary = build_cohorts(manifest, config); outer, inner = build_splits(manifest, cohorts, config); validate_design(manifest, cohorts, outer, inner)
    outputs = config["outputs"]; split_dir, table_dir, figure_dir, report_dir = Path(outputs["split_directory"]), Path(outputs["table_directory"]), Path(outputs["figure_directory"]), Path(outputs["report_directory"]); progress_root, checkpoint_root = Path(outputs["progress_directory"]), Path(outputs["checkpoint_directory"])
    for directory in (split_dir, table_dir, figure_dir, report_dir, progress_root, checkpoint_root): directory.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(split_dir / "stage_17_cohort_assignments.csv", index=False); outer.to_csv(split_dir / "stage_17_outer_fold_assignments.csv", index=False); inner.to_csv(split_dir / "stage_17_inner_fold_assignments.csv", index=False); cohort_summary.to_csv(table_dir / "stage_17_cohort_summary.csv", index=False)
    if dry_run:
        return {"dry_run": True, "cohorts": len(cohort_summary), "cells": sorted(cohort_summary["actual_cells"].unique().tolist()), "models": len(MODEL_LABELS)}
    stage15_base, stage16_base = load_yaml_config(config["stage_15_config"]), load_yaml_config(config["stage_16_config"])
    peptide, protein = pd.read_parquet(config["data"]["peptide_matrix"]), pd.read_parquet(config["data"]["protein_matrix"]); directory = Path(config["data"]["directory"]); spectrum_bags, spectrum_metadata = pd.read_parquet(directory / "spectrum_bag_index.parquet"), pd.read_parquet(directory / "spectrum_metadata.parquet")
    config15_for_bags = copy.deepcopy(stage15_base); config15_for_bags["data"]["peptide_bag_directory"] = config["data"]["peptide_bag_directory"]; peptide_vocabulary, peptide_instances, peptide_bags = make_peptide_bags(config15_for_bags)
    all_results: dict[str, list[pd.DataFrame]] = {name: [] for name in ("predictions", "fold_metrics", "runtime", "tuning", "history")}; logger = logging.getLogger("single_cell_proteomics_mil"); run_started = time.perf_counter()
    only_cohort = os.environ.get("STAGE17_ONLY_COHORT")
    selected_summary = cohort_summary[cohort_summary["cohort_id"].eq(only_cohort)] if only_cohort else cohort_summary
    if only_cohort and selected_summary.empty:
        raise ValueError(f"unknown Stage 17 cohort: {only_cohort}")
    for row in selected_summary.sort_values(["actual_cells", "repeat"]).itertuples(index=False):
        context = {"cohort_id": row.cohort_id, "target_cells": int(row.target_cells), "actual_cells": int(row.actual_cells), "repeat": int(row.repeat)}; progress_dir = progress_root / row.cohort_id
        logger.info("Stage 17 | cohort=%s cells=%s groups=%s", row.cohort_id, row.actual_cells, row.raw_file_groups)
        selected_ids = cohorts[cohorts["cohort_id"].eq(row.cohort_id)]["sample_id"]; cohort_manifest = manifest[manifest["sample_id"].isin(selected_ids)].copy(); cohort_outer = outer[outer["cohort_id"].eq(row.cohort_id)].drop(columns=["cohort_id", "target_cells", "repeat"]); cohort_inner = inner[inner["cohort_id"].eq(row.cohort_id)].drop(columns=["cohort_id", "target_cells", "repeat"])
        if int(row.actual_cells) == len(manifest) and int(row.repeat) == 0:
            results = _reuse_full_corpus(context); progress_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in results.items(): frame.to_parquet(progress_dir / f"{name}.parquet", index=False)
            (progress_dir / "complete.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
        else:
            results = run_cohort(cohort_manifest, peptide, protein, spectrum_bags, spectrum_metadata, peptide_vocabulary, peptide_instances, peptide_bags, cohort_outer, cohort_inner, config, stage15_base, stage16_base, context, progress_dir, checkpoint_root / row.cohort_id)
        for name, frame in results.items(): all_results[name].append(frame)
    if only_cohort:
        return {"worker": True, "cohort_id": only_cohort, "completed": True}
    combined = {name: pd.concat(frames, ignore_index=True, sort=False) for name, frames in all_results.items()}
    combined["predictions"].to_parquet(table_dir / "stage_17_predictions.parquet", index=False); combined["tuning"].to_parquet(table_dir / "stage_17_tuning.parquet", index=False); combined["history"].to_parquet(table_dir / "stage_17_training_history.parquet", index=False)
    fold_metrics = add_fold_counts(combined["fold_metrics"], outer); fold_metrics.to_csv(table_dir / "stage_17_fold_seed_metrics.csv", index=False)
    repeat_metrics, intervals, paired, bootstraps, averaged = evaluate_learning_curves(combined["predictions"], manifest, config)
    repeat_metrics.to_csv(table_dir / "stage_17_repeat_metrics.csv", index=False); intervals.to_csv(table_dir / "stage_17_bootstrap_intervals.csv", index=False); paired.to_csv(table_dir / "stage_17_paired_differences.csv", index=False); bootstraps.to_parquet(table_dir / "stage_17_bootstrap_replicates.parquet", index=False); averaged.to_parquet(table_dir / "stage_17_averaged_predictions.parquet", index=False)
    learning = summarize_learning_curves(repeat_metrics, intervals, cohort_summary); learning.to_csv(table_dir / "stage_17_learning_curves.csv", index=False)
    compute = compute_scaling_table(combined["runtime"]); compute.to_csv(table_dir / "stage_17_compute_scaling.csv", index=False); make_figures(learning, compute, figure_dir)
    summary = {"targets": [int(value) for value in config["target_cells"]], "maximum_cells": len(manifest), "cohorts": len(cohort_summary), "subsampling_repeats": int(config["subsampling_repeats"]), "models": len(MODEL_LABELS), "bootstrap_replicates": int(config["bootstrap_replicates"]), "prediction_rows": len(combined["predictions"]), "fold_metric_rows": len(fold_metrics), "runtime_hours": float(combined["runtime"]["runtime_seconds"].sum() / 3600), "wall_clock_hours": (time.perf_counter() - run_started) / 3600}
    (report_dir / "stage_17_learning_curves_data_sufficiency.md").write_text(render_report(summary, learning), encoding="utf-8"); (table_dir / "stage_17_run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="configs/stage_17.yaml"); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_17_learning_curves.log")); result = run_stage_seventeen(args.config, args.dry_run); logging.getLogger("single_cell_proteomics_mil").info("Stage 17 completed: %s", result)


if __name__ == "__main__":
    main()
