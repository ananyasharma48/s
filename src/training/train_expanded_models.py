"""Execute Stage 15 expanded-corpus retraining and grouped evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from torch import nn

from src.data.build_peptide_bags import (
    encode_peptide_bags,
    make_peptide_vocabulary,
    prepare_inputs,
    validate_bag_dataset,
)
from src.evaluation.bootstrap import (
    DEFAULT_METRICS,
    group_bootstrap_metrics,
    paired_group_bootstrap_differences,
    percentile_interval,
)
from src.evaluation.calibration import calibration_metrics, reliability_table
from src.evaluation.metrics import binary_metrics
from src.models.classical import batch_logistic, fixed_vector_logistic, peptide_random_forest
from src.training.datasets import PeptideBagDataset, fit_intensity_standardizer, validate_mil_tables
from src.training.spectrum_datasets import (
    SpectrumArrayStore,
    SpectrumBagDataset,
    fit_numeric_standardizer,
    make_psm_vocabulary,
    validate_spectrum_bags,
)
from src.training.train_baseline import encode_labels, predefined_inner_split
from src.training.train_mil import (
    build_model as build_peptide_model,
    evaluate_model as evaluate_peptide_model,
    fit_final_model as fit_final_peptide_model,
    make_loader as make_peptide_loader,
    positive_class_weight as peptide_class_weight,
    resolve_device,
    select_epoch_count as select_peptide_epochs,
)
from src.training.train_spectrum_mil import (
    VARIANT_LABELS,
    build_spectrum_model,
    checkpoint_payload as spectrum_checkpoint_payload,
    evaluate_repeated as evaluate_spectrum_repeated,
    fit_final_model as fit_final_spectrum_model,
    peak_memory_mib,
    positive_class_weight as spectrum_class_weight,
    select_epoch_count as select_spectrum_epochs,
    variant_options,
)
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging
from src.utils.seeds import seed_everything


MODEL_LABELS = {
    "majority": "Majority",
    "batch_logistic": "Batch-only logistic",
    "protein_logistic": "Protein logistic",
    "peptide_logistic": "Peptide logistic",
    "peptide_random_forest": "Peptide random forest",
    "mean_pooling": "Mean-pooling peptide MIL",
    "max_pooling": "Max-pooling peptide MIL",
    "attention": "Attention peptide MIL",
    "reporter_only": VARIANT_LABELS["reporter_only"],
    "spectrum_embedding": VARIANT_LABELS["spectrum_embedding"],
    "spectrum_psm": VARIANT_LABELS["spectrum_psm"],
}
MODEL_FAMILIES = {
    **{name: "conventional" for name in (
        "majority", "batch_logistic", "protein_logistic", "peptide_logistic",
        "peptide_random_forest",
    )},
    **{name: "peptide_mil" for name in ("mean_pooling", "max_pooling", "attention")},
    **{name: "spectrum_mil" for name in (
        "reporter_only", "spectrum_embedding", "spectrum_psm",
    )},
}
METRIC_COLUMNS = [
    "balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc",
    "sensitivity", "specificity", "accuracy",
]
PROGRESS_NAMES = ("predictions", "fold_metrics", "history", "runtime", "tuning")


def resident_memory_mib() -> float:
    """Return process peak resident memory using the platform convention."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def append_records(target: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    target.extend(records)


def load_progress(table_directory: Path, enabled: bool) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in PROGRESS_NAMES}
    if not enabled:
        return output
    for name in PROGRESS_NAMES:
        path = table_directory / f"stage_15_{name}.partial.csv"
        if path.exists() and path.stat().st_size:
            output[name] = pd.read_csv(path).to_dict("records")
    return output


def save_progress(table_directory: Path, progress: dict[str, list[dict[str, Any]]]) -> None:
    for name in PROGRESS_NAMES:
        frame = pd.DataFrame(progress[name])
        if not frame.empty:
            frame.to_csv(table_directory / f"stage_15_{name}.partial.csv", index=False)


def experiment_complete(
    progress: dict[str, list[dict[str, Any]]], model: str, seed: int,
    outer_fold: int, expected_samples: int,
) -> bool:
    frame = pd.DataFrame(progress["predictions"])
    if frame.empty:
        return False
    selected = frame[
        frame["model"].eq(model)
        & frame["seed"].eq(seed)
        & frame["outer_fold"].eq(outer_fold)
    ]
    return len(selected) == expected_samples and selected["sample_id"].nunique() == expected_samples


def make_peptide_bags(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create or validate the Stage 15 peptide-MIL representation."""
    data = config["data"]
    output = Path(data["peptide_bag_directory"])
    paths = {
        "vocabulary": output / "peptide_vocabulary.parquet",
        "instances": output / "peptide_instances.parquet",
        "bags": output / "bag_index.parquet",
    }
    if all(path.exists() for path in paths.values()):
        vocabulary = pd.read_parquet(paths["vocabulary"])
        instances = pd.read_parquet(paths["instances"])
        bags = pd.read_parquet(paths["bags"])
        validate_mil_tables(instances, bags, vocabulary)
        return vocabulary, instances, bags
    matrix = pd.read_parquet(data["peptide_matrix"])
    manifest = pd.read_parquet(data["manifest"])
    matrix, valid = prepare_inputs(matrix, manifest, str(config["positive_label"]))
    vocabulary = make_peptide_vocabulary(matrix)
    instances, bags = encode_peptide_bags(matrix, valid, vocabulary)
    validate_bag_dataset(matrix, instances, bags, vocabulary)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary.to_parquet(paths["vocabulary"], index=False)
    instances.to_parquet(paths["instances"], index=False)
    bags.to_parquet(paths["bags"], index=False)
    return vocabulary, instances, bags


def validate_inputs(
    manifest: pd.DataFrame, peptide: pd.DataFrame, protein: pd.DataFrame,
    spectrum_bags: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame,
) -> None:
    ids = set(manifest["sample_id"].astype(str))
    if len(ids) != len(manifest) or ids != set(peptide.columns.astype(str)):
        raise ValueError("manifest and peptide matrix are not aligned")
    if ids != set(protein.columns.astype(str)) or ids != set(spectrum_bags["sample_id"].astype(str)):
        raise ValueError("manifest, protein matrix, and spectrum bags are not aligned")
    test_counts = outer.loc[outer["partition"].eq("test"), "sample_id"].value_counts()
    if set(test_counts.index) != ids or not test_counts.eq(1).all():
        raise ValueError("every cell must occur in exactly one grouped outer test fold")
    for fold, frame in outer.groupby("outer_fold"):
        if frame.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
            raise ValueError(f"raw-file leakage in outer fold {fold}")
        train_ids = set(frame.loc[frame["partition"].eq("train"), "sample_id"])
        selected = inner[inner["outer_fold"].eq(fold)]
        for inner_fold, inner_frame in selected.groupby("inner_fold"):
            if set(inner_frame["sample_id"]) != train_ids:
                raise ValueError(f"inner fold {fold}/{inner_fold} does not cover outer training cells")
            if inner_frame.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
                raise ValueError(f"raw-file leakage in inner fold {fold}/{inner_fold}")


def add_predictions(
    target: list[dict[str, Any]], model: str, family: str, seed: int,
    outer_fold: int, sample_ids: list[str], labels: pd.Series,
    y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray,
) -> None:
    for sample_id, label, truth, prediction, probability in zip(
        sample_ids, labels, y_true, predictions, probabilities
    ):
        target.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "model_family": family,
            "seed": int(seed),
            "outer_fold": int(outer_fold),
            "sample_id": str(sample_id),
            "label_raw": str(label),
            "y_true": int(truth),
            "y_pred": int(prediction),
            "probability_sc_u": float(probability),
        })


def add_fold_metric(
    target: list[dict[str, Any]], model: str, seed: int, outer_fold: int,
    sample_count: int, selected_epochs: int | None, best_inner_score: float | None,
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray,
) -> None:
    target.append({
        "model": model,
        "model_label": MODEL_LABELS[model],
        "model_family": MODEL_FAMILIES[model],
        "seed": int(seed),
        "outer_fold": int(outer_fold),
        "test_samples": int(sample_count),
        "selected_epochs": selected_epochs,
        "best_inner_balanced_accuracy": best_inner_score,
        **binary_metrics(y_true, y_pred, probabilities),
    })


def train_conventional(
    config: dict[str, Any], manifest: pd.DataFrame, peptide: pd.DataFrame,
    protein: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame,
    checkpoint_directory: Path, table_directory: Path,
    progress: dict[str, list[dict[str, Any]]],
) -> None:
    logger = logging.getLogger("single_cell_proteomics_mil")
    seed = int(config["seed"])
    positive = str(config["positive_label"])
    settings = config["conventional"]
    metadata = manifest.set_index("sample_id")
    batch_fields = ["batch_chromatography", "batch_sort", "batch_digest"]
    for outer_fold in sorted(outer["outer_fold"].unique()):
        fold = outer[outer["outer_fold"].eq(outer_fold)]
        train_ids = fold.loc[fold["partition"].eq("train"), "sample_id"].tolist()
        test_ids = fold.loc[fold["partition"].eq("test"), "sample_id"].tolist()
        train_labels = metadata.loc[train_ids, "label_raw"]
        test_labels = metadata.loc[test_ids, "label_raw"]
        y_train = encode_labels(train_labels, positive)
        y_test = encode_labels(test_labels, positive)
        inner_split = predefined_inner_split(inner, int(outer_fold), train_ids)
        for model_name in (
            "majority", "batch_logistic", "protein_logistic", "peptide_logistic",
            "peptide_random_forest",
        ):
            if experiment_complete(progress, model_name, seed, int(outer_fold), len(test_ids)):
                continue
            started = time.perf_counter()
            logger.info("Stage 15 | fold=%s model=%s", outer_fold, model_name)
            tuning: dict[str, Any] = {}
            if model_name == "majority":
                probability = float(y_train.mean())
                fitted: Any = {"predicted_class": int(probability >= 0.5), "probability": probability}
                predictions = np.full(len(test_ids), int(probability >= 0.5), dtype=int)
                probabilities = np.full(len(test_ids), probability, dtype=float)
            elif model_name == "batch_logistic":
                search = GridSearchCV(
                    batch_logistic(seed),
                    {"model__C": [float(value) for value in settings["logistic_c_grid"]]},
                    scoring="balanced_accuracy", cv=inner_split, refit=True, n_jobs=1,
                )
                search.fit(metadata.loc[train_ids, batch_fields], y_train)
                fitted = search.best_estimator_
                predictions = fitted.predict(metadata.loc[test_ids, batch_fields])
                probabilities = fitted.predict_proba(metadata.loc[test_ids, batch_fields])[:, 1]
                tuning = {"selected_c": search.best_params_["model__C"], "best_inner_balanced_accuracy": search.best_score_}
            elif model_name in {"protein_logistic", "peptide_logistic"}:
                matrix = protein if model_name == "protein_logistic" else peptide
                search = GridSearchCV(
                    fixed_vector_logistic(float(settings["minimum_feature_prevalence"]), seed),
                    {"model__C": [float(value) for value in settings["logistic_c_grid"]]},
                    scoring="balanced_accuracy", cv=inner_split, refit=True, n_jobs=1,
                )
                search.fit(matrix.reindex(columns=train_ids).T, y_train)
                fitted = search.best_estimator_
                predictions = fitted.predict(matrix.reindex(columns=test_ids).T)
                probabilities = fitted.predict_proba(matrix.reindex(columns=test_ids).T)[:, 1]
                tuning = {"selected_c": search.best_params_["model__C"], "best_inner_balanced_accuracy": search.best_score_}
            else:
                forest = settings["random_forest"]
                fitted = peptide_random_forest(
                    float(settings["minimum_feature_prevalence"]), seed + int(outer_fold),
                    int(forest["n_estimators"]), int(forest["min_samples_leaf"]),
                )
                fitted.fit(peptide.reindex(columns=train_ids).T, y_train)
                predictions = fitted.predict(peptide.reindex(columns=test_ids).T)
                probabilities = fitted.predict_proba(peptide.reindex(columns=test_ids).T)[:, 1]
            joblib.dump(fitted, checkpoint_directory / f"{model_name}_seed_{seed}_fold_{outer_fold}.joblib")
            add_predictions(
                progress["predictions"], model_name, "conventional", seed, int(outer_fold),
                test_ids, test_labels, y_test, np.asarray(predictions), np.asarray(probabilities),
            )
            add_fold_metric(
                progress["fold_metrics"], model_name, seed, int(outer_fold), len(test_ids),
                None, tuning.get("best_inner_balanced_accuracy"), y_test,
                np.asarray(predictions), np.asarray(probabilities),
            )
            progress["tuning"].append({
                "model": model_name, "seed": seed, "outer_fold": int(outer_fold),
                "selection_method": "three grouped inner folds" if tuning else "prespecified",
                "decision_threshold": 0.5, **tuning,
            })
            progress["runtime"].append({
                "model": model_name, "model_family": "conventional", "seed": seed,
                "outer_fold": int(outer_fold), "runtime_seconds": time.perf_counter() - started,
                "process_peak_memory_mib": resident_memory_mib(),
            })
            save_progress(table_directory, progress)


def train_peptide_mil(
    config: dict[str, Any], vocabulary: pd.DataFrame, instances: pd.DataFrame,
    bags: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame,
    checkpoint_directory: Path, table_directory: Path,
    progress: dict[str, list[dict[str, Any]]], device: torch.device,
) -> None:
    logger = logging.getLogger("single_cell_proteomics_mil")
    settings = config["peptide_mil"]
    training = dict(settings["training"])
    vocabulary_size = int(vocabulary["peptide_id"].max()) + 1
    label_by_sample = bags.set_index("sample_id")["label_raw"]
    validation_inner_fold = int(config["validation_inner_fold"])
    for outer_fold in sorted(outer["outer_fold"].unique()):
        fold = outer[outer["outer_fold"].eq(outer_fold)]
        outer_train = fold.loc[fold["partition"].eq("train"), "sample_id"].tolist()
        test_ids = fold.loc[fold["partition"].eq("test"), "sample_id"].tolist()
        selected = inner[
            inner["outer_fold"].eq(outer_fold)
            & inner["inner_fold"].eq(validation_inner_fold)
        ]
        selection_ids = selected.loc[selected["partition"].eq("train"), "sample_id"].tolist()
        validation_ids = selected.loc[selected["partition"].eq("validation"), "sample_id"].tolist()
        for model_index, model_name in enumerate(settings["models"]):
            for seed in config["neural_seeds"]:
                seed = int(seed)
                if experiment_complete(progress, model_name, seed, int(outer_fold), len(test_ids)):
                    continue
                started = time.perf_counter()
                run_seed = seed + int(outer_fold) * 1_000 + model_index * 10_000
                logger.info("Stage 15 | fold=%s seed=%s peptide_mil=%s", outer_fold, seed, model_name)
                seed_everything(run_seed)
                selection_standardizer = fit_intensity_standardizer(instances, bags, selection_ids)
                training_dataset = PeptideBagDataset(
                    instances, bags, selection_ids, selection_standardizer,
                    int(training["maximum_training_instances_per_bag"]), run_seed,
                )
                validation_dataset = PeptideBagDataset(
                    instances, bags, validation_ids, selection_standardizer, None, run_seed,
                )
                selection_model = build_peptide_model(model_name, vocabulary_size, settings["model"]).to(device)
                selection_checkpoint = checkpoint_directory / f"{model_name}_seed_{seed}_fold_{outer_fold}_selection.pt"
                best_epoch, best_score, history = select_peptide_epochs(
                    model_name, selection_model, training_dataset, validation_dataset,
                    selection_ids, validation_ids, bags, selection_standardizer,
                    int(outer_fold), run_seed, vocabulary_size, settings["model"], training,
                    selection_checkpoint, device,
                )
                for record in history:
                    record.update({"seed": seed, "model_family": "peptide_mil"})
                append_records(progress["history"], history)
                seed_everything(run_seed)
                final_standardizer = fit_intensity_standardizer(instances, bags, outer_train)
                final_dataset = PeptideBagDataset(
                    instances, bags, outer_train, final_standardizer,
                    int(training["maximum_training_instances_per_bag"]), run_seed,
                )
                test_dataset = PeptideBagDataset(
                    instances, bags, test_ids, final_standardizer, None, run_seed,
                )
                final_model = build_peptide_model(model_name, vocabulary_size, settings["model"]).to(device)
                final_history = fit_final_peptide_model(
                    model_name, final_model, final_dataset, outer_train, bags, best_epoch,
                    int(outer_fold), run_seed, training, device,
                )
                for record in final_history:
                    record.update({"seed": seed, "model_family": "peptide_mil"})
                append_records(progress["history"], final_history)
                criterion = nn.BCEWithLogitsLoss(pos_weight=peptide_class_weight(bags, outer_train, device))
                test = evaluate_peptide_model(
                    final_model,
                    make_peptide_loader(test_dataset, int(training["batch_size"]), False, run_seed),
                    criterion, device, collect_attention=False,
                )
                torch.save({
                    "stage": 15, "model_name": model_name, "seed": seed,
                    "training_seed": run_seed, "outer_fold": int(outer_fold),
                    "epochs": best_epoch, "model_config": settings["model"],
                    "intensity_standardizer": final_standardizer.as_dict(),
                    "training_sample_ids": outer_train, "test_sample_ids": test_ids,
                    "model_state_dict": final_model.state_dict(),
                }, checkpoint_directory / f"{model_name}_seed_{seed}_fold_{outer_fold}_final.pt")
                add_predictions(
                    progress["predictions"], model_name, "peptide_mil", seed,
                    int(outer_fold), test["sample_ids"], label_by_sample.loc[test["sample_ids"]],
                    test["y_true"], test["y_pred"], test["probabilities"],
                )
                add_fold_metric(
                    progress["fold_metrics"], model_name, seed, int(outer_fold), len(test_ids),
                    best_epoch, best_score, test["y_true"], test["y_pred"], test["probabilities"],
                )
                progress["tuning"].append({
                    "model": model_name, "seed": seed, "outer_fold": int(outer_fold),
                    "selection_method": f"grouped inner fold {validation_inner_fold}",
                    "selected_epochs": best_epoch,
                    "best_inner_balanced_accuracy": best_score,
                    "decision_threshold": 0.5,
                })
                progress["runtime"].append({
                    "model": model_name, "model_family": "peptide_mil", "seed": seed,
                    "outer_fold": int(outer_fold), "runtime_seconds": time.perf_counter() - started,
                    "process_peak_memory_mib": resident_memory_mib(),
                })
                save_progress(table_directory, progress)


def train_spectrum_mil(
    config: dict[str, Any], bags: pd.DataFrame, metadata: pd.DataFrame,
    spectra_path: Path, reporter_path: Path, outer: pd.DataFrame, inner: pd.DataFrame,
    checkpoint_directory: Path, table_directory: Path,
    progress: dict[str, list[dict[str, Any]]], device: torch.device,
) -> None:
    logger = logging.getLogger("single_cell_proteomics_mil")
    settings = config["spectrum_mil"]
    training = dict(settings["training"])
    model_config = settings["model"]
    peptide_to_id, vocabulary = make_psm_vocabulary(metadata)
    vocabulary_size = int(vocabulary["peptide_id"].max()) + 1 if len(vocabulary) else 1
    with __import__("h5py").File(spectra_path, "r") as handle:
        source_spectrum_dim = int(handle["spectrum_embedding"].shape[1])
    array_store = SpectrumArrayStore.from_hdf5(
        spectra_path, reporter_path, int(model_config["spectrum_downsample_factor"])
    )
    label_by_sample = bags.set_index("sample_id")["label_raw"]
    validation_inner_fold = int(config["validation_inner_fold"])
    for outer_fold in sorted(outer["outer_fold"].unique()):
        fold = outer[outer["outer_fold"].eq(outer_fold)]
        outer_train = fold.loc[fold["partition"].eq("train"), "sample_id"].tolist()
        test_ids = fold.loc[fold["partition"].eq("test"), "sample_id"].tolist()
        selected = inner[
            inner["outer_fold"].eq(outer_fold)
            & inner["inner_fold"].eq(validation_inner_fold)
        ]
        selection_ids = selected.loc[selected["partition"].eq("train"), "sample_id"].tolist()
        validation_ids = selected.loc[selected["partition"].eq("validation"), "sample_id"].tolist()
        for variant_index, variant in enumerate(settings["variants"]):
            use_spectrum, use_psm, _ = variant_options(variant)
            for seed in config["neural_seeds"]:
                seed = int(seed)
                if experiment_complete(progress, variant, seed, int(outer_fold), len(test_ids)):
                    continue
                started = time.perf_counter()
                run_seed = seed + int(outer_fold) * 1_000 + variant_index * 10_000
                logger.info("Stage 15 | fold=%s seed=%s spectrum_mil=%s", outer_fold, seed, variant)
                seed_everything(run_seed)
                selection_standardizer = fit_numeric_standardizer(
                    bags, metadata, selection_ids, spectra_path, reporter_path, use_psm, array_store,
                )
                selection_dataset = SpectrumBagDataset(
                    bags, metadata, selection_ids, spectra_path, reporter_path,
                    selection_standardizer, peptide_to_id, use_spectrum, use_psm,
                    int(model_config["spectrum_downsample_factor"]),
                    int(training["maximum_training_instances_per_bag"]), run_seed, array_store,
                )
                validation_dataset = SpectrumBagDataset(
                    bags, metadata, validation_ids, spectra_path, reporter_path,
                    selection_standardizer, peptide_to_id, use_spectrum, use_psm,
                    int(model_config["spectrum_downsample_factor"]),
                    int(training["maximum_evaluation_instances_per_bag"]), run_seed, array_store,
                )
                selection_model = build_spectrum_model(
                    variant, source_spectrum_dim, vocabulary_size, model_config
                ).to(device)
                best_epoch, best_score, history, best_state = select_spectrum_epochs(
                    variant, selection_model, selection_dataset, validation_dataset,
                    bags, selection_ids, training, device,
                )
                for record in history:
                    record.update({
                        "model": variant, "seed": seed, "outer_fold": int(outer_fold),
                        "model_family": "spectrum_mil",
                    })
                append_records(progress["history"], history)
                torch.save(
                    spectrum_checkpoint_payload(
                        variant, variant, selection_model, int(outer_fold), run_seed,
                        best_epoch, selection_standardizer, selection_ids, model_config,
                    ) | {
                        "stage": 15, "seed": seed, "validation_sample_ids": validation_ids,
                        "best_validation_balanced_accuracy": best_score,
                        "model_state_dict": best_state,
                    },
                    checkpoint_directory / f"{variant}_seed_{seed}_fold_{outer_fold}_selection.pt",
                )
                selection_dataset.close()
                validation_dataset.close()
                seed_everything(run_seed)
                final_standardizer = fit_numeric_standardizer(
                    bags, metadata, outer_train, spectra_path, reporter_path, use_psm, array_store,
                )
                final_dataset = SpectrumBagDataset(
                    bags, metadata, outer_train, spectra_path, reporter_path,
                    final_standardizer, peptide_to_id, use_spectrum, use_psm,
                    int(model_config["spectrum_downsample_factor"]),
                    int(training["maximum_training_instances_per_bag"]), run_seed, array_store,
                )
                final_model = build_spectrum_model(
                    variant, source_spectrum_dim, vocabulary_size, model_config
                ).to(device)
                final_history = fit_final_spectrum_model(
                    variant, final_model, final_dataset, bags, outer_train,
                    best_epoch, training, device,
                )
                for record in final_history:
                    record.update({
                        "model": variant, "seed": seed, "outer_fold": int(outer_fold),
                        "model_family": "spectrum_mil",
                    })
                append_records(progress["history"], final_history)
                final_dataset.close()
                test_dataset = SpectrumBagDataset(
                    bags, metadata, test_ids, spectra_path, reporter_path,
                    final_standardizer, peptide_to_id, use_spectrum, use_psm,
                    int(model_config["spectrum_downsample_factor"]),
                    int(training["maximum_evaluation_instances_per_bag"]), run_seed, array_store,
                )
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=spectrum_class_weight(bags, outer_train, device)
                )
                test = evaluate_spectrum_repeated(
                    final_model, test_dataset, criterion, device,
                    int(training["evaluation_batch_size"]), run_seed,
                    int(training["evaluation_repeats"]), False,
                )
                test_dataset.close()
                torch.save(
                    spectrum_checkpoint_payload(
                        variant, variant, final_model, int(outer_fold), run_seed,
                        best_epoch, final_standardizer, outer_train, model_config,
                    ) | {"stage": 15, "seed": seed, "test_sample_ids": test_ids},
                    checkpoint_directory / f"{variant}_seed_{seed}_fold_{outer_fold}_final.pt",
                )
                add_predictions(
                    progress["predictions"], variant, "spectrum_mil", seed,
                    int(outer_fold), test["sample_ids"], label_by_sample.loc[test["sample_ids"]],
                    test["y_true"], test["y_pred"], test["probabilities"],
                )
                add_fold_metric(
                    progress["fold_metrics"], variant, seed, int(outer_fold), len(test_ids),
                    best_epoch, best_score, test["y_true"], test["y_pred"], test["probabilities"],
                )
                progress["tuning"].append({
                    "model": variant, "seed": seed, "outer_fold": int(outer_fold),
                    "selection_method": f"grouped inner fold {validation_inner_fold}",
                    "selected_epochs": best_epoch,
                    "best_inner_balanced_accuracy": best_score,
                    "decision_threshold": 0.5,
                })
                progress["runtime"].append({
                    "model": variant, "model_family": "spectrum_mil", "seed": seed,
                    "outer_fold": int(outer_fold), "runtime_seconds": time.perf_counter() - started,
                    "process_peak_memory_mib": peak_memory_mib(),
                })
                save_progress(table_directory, progress)


def validate_predictions(
    predictions: pd.DataFrame, expected_ids: set[str], conventional_seed: int,
    neural_seeds: list[int],
) -> None:
    expected_models = set(MODEL_LABELS)
    if set(predictions["model"]) != expected_models:
        raise ValueError("Stage 15 predictions do not contain every required model")
    for model, frame in predictions.groupby("model"):
        seeds = [conventional_seed] if MODEL_FAMILIES[model] == "conventional" else neural_seeds
        if set(frame["seed"].astype(int)) != set(seeds):
            raise ValueError(f"{model} has incorrect seed coverage")
        for seed, seeded in frame.groupby("seed"):
            if len(seeded) != len(expected_ids) or set(seeded["sample_id"]) != expected_ids:
                raise ValueError(f"{model} seed {seed} lacks one OOF prediction per cell")
            if seeded["sample_id"].duplicated().any():
                raise ValueError(f"{model} seed {seed} contains duplicate cells")
            probabilities = seeded["probability_sc_u"].to_numpy(float)
            if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError(f"{model} seed {seed} has invalid probabilities")


def averaged_predictions(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Average neural seeds into one paired OOF prediction per model and cell."""
    keys = ["model", "model_label", "model_family", "sample_id", "label_raw", "y_true", "outer_fold"]
    averaged = predictions.groupby(keys, as_index=False)["probability_sc_u"].mean()
    averaged["y_pred"] = averaged["probability_sc_u"].ge(0.5).astype(int)
    extra = metadata[[
        "sample_id", "raw_file_stem", "source_cohort", "batch_chromatography",
        "batch_sort", "batch_digest", "reporter_channel",
    ]].drop_duplicates("sample_id")
    return averaged.merge(extra, on="sample_id", how="left", validate="many_to_one")


def overall_seed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), frame in predictions.groupby(["model", "seed"]):
        rows.append({
            "model": model, "model_label": MODEL_LABELS[model],
            "model_family": MODEL_FAMILIES[model], "seed": int(seed), "samples": len(frame),
            **binary_metrics(
                frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int),
                frame["probability_sc_u"].to_numpy(float),
            ),
            **calibration_metrics(
                frame["y_true"].to_numpy(int), frame["probability_sc_u"].to_numpy(float), 10,
            ),
        })
    return pd.DataFrame(rows)


def subgroup_metrics(averaged: pd.DataFrame, minimum_samples: int) -> pd.DataFrame:
    rows = []
    for model, model_frame in averaged.groupby("model"):
        for grouping in ("batch_chromatography", "source_cohort"):
            for group, frame in model_frame.groupby(grouping, dropna=False):
                truths = frame["y_true"].to_numpy(int)
                record: dict[str, Any] = {
                    "model": model, "model_label": MODEL_LABELS[model],
                    "grouping_variable": grouping, "group_value": group,
                    "samples": len(frame), "negative_samples": int((truths == 0).sum()),
                    "positive_samples": int((truths == 1).sum()),
                }
                if len(frame) >= minimum_samples and np.unique(truths).size == 2:
                    record.update(binary_metrics(
                        truths, frame["y_pred"].to_numpy(int),
                        frame["probability_sc_u"].to_numpy(float),
                    ))
                else:
                    record.update({metric: np.nan for metric in METRIC_COLUMNS})
                rows.append(record)
    return pd.DataFrame(rows)


def evaluate_models(
    config: dict[str, Any], predictions: pd.DataFrame, manifest: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    evaluation = config["evaluation"]
    averaged = averaged_predictions(predictions, manifest)
    seed_metrics = overall_seed_metrics(predictions)
    averaged_metric_rows = []
    for model, frame in averaged.groupby("model"):
        averaged_metric_rows.append({
            "model": model,
            **{
                f"averaged_seed_{key}": value
                for key, value in (
                    binary_metrics(
                        frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int),
                        frame["probability_sc_u"].to_numpy(float),
                    )
                    | calibration_metrics(
                        frame["y_true"].to_numpy(int),
                        frame["probability_sc_u"].to_numpy(float),
                        int(evaluation["calibration_bins"]),
                    )
                ).items()
            },
        })
    averaged_metrics = pd.DataFrame(averaged_metric_rows)
    subgroups = subgroup_metrics(averaged, int(evaluation["minimum_subgroup_samples"]))
    bootstrap_frames, intervals = [], []
    paired_frames, paired_rows = [], []
    replicate_count = int(evaluation["bootstrap_replicates"])
    confidence = float(evaluation["confidence_level"])
    group_column = str(evaluation["bootstrap_group_column"])
    seed = int(config["seed"])
    for model_index, (model, frame) in enumerate(averaged.groupby("model", sort=True)):
        point = binary_metrics(
            frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int),
            frame["probability_sc_u"].to_numpy(float),
        )
        bootstrap = group_bootstrap_metrics(
            frame, group_column, replicate_count, seed + model_index * 10_000
        )
        bootstrap.insert(0, "model", model)
        bootstrap_frames.append(bootstrap)
        for metric in DEFAULT_METRICS:
            lower, upper = percentile_interval(bootstrap[metric].to_numpy(), confidence)
            intervals.append({
                "model": model, "model_label": MODEL_LABELS[model], "metric": metric,
                "point_estimate": point[metric], "ci_lower": lower, "ci_upper": upper,
                "confidence_level": confidence, "bootstrap_replicates": replicate_count,
                "bootstrap_group": group_column,
            })
    reference_name = str(evaluation["reference_model"])
    reference = averaged[averaged["model"].eq(reference_name)]
    reference_point = binary_metrics(
        reference["y_true"].to_numpy(int), reference["y_pred"].to_numpy(int),
        reference["probability_sc_u"].to_numpy(float),
    )
    for model_index, model in enumerate(name for name in MODEL_LABELS if name != reference_name):
        comparator = averaged[averaged["model"].eq(model)]
        comparator_point = binary_metrics(
            comparator["y_true"].to_numpy(int), comparator["y_pred"].to_numpy(int),
            comparator["probability_sc_u"].to_numpy(float),
        )
        paired = paired_group_bootstrap_differences(
            reference, comparator, group_column, replicate_count,
            seed + 200_000 + model_index * 10_000,
        )
        paired.insert(0, "comparator_model", model)
        paired.insert(0, "reference_model", reference_name)
        paired_frames.append(paired)
        for metric in DEFAULT_METRICS:
            lower, upper = percentile_interval(paired[metric].to_numpy(), confidence)
            paired_rows.append({
                "reference_model": reference_name, "comparator_model": model,
                "metric": metric,
                "point_difference_comparator_minus_reference": comparator_point[metric] - reference_point[metric],
                "ci_lower": lower, "ci_upper": upper, "confidence_level": confidence,
                "bootstrap_replicates": replicate_count, "bootstrap_group": group_column,
            })
    reliability_frames = []
    for model, frame in averaged.groupby("model"):
        reliability = reliability_table(
            frame["y_true"].to_numpy(int), frame["probability_sc_u"].to_numpy(float),
            int(evaluation["calibration_bins"]),
        )
        reliability.insert(0, "model", model)
        reliability_frames.append(reliability)
    interval_frame = pd.DataFrame(intervals)
    model_summary = seed_metrics.groupby(
        ["model", "model_label", "model_family"], as_index=False
    )[METRIC_COLUMNS + ["brier_score", "expected_calibration_error"]].agg(["mean", "std"])
    model_summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in model_summary.columns
    ]
    model_summary = model_summary.reset_index(drop=True)
    model_summary["seeds"] = model_summary["model"].map(
        seed_metrics.groupby("model")["seed"].nunique()
    )
    ba_intervals = interval_frame[interval_frame["metric"].eq("balanced_accuracy")][
        ["model", "point_estimate", "ci_lower", "ci_upper"]
    ].rename(columns={
        "point_estimate": "averaged_seed_balanced_accuracy",
        "ci_lower": "balanced_accuracy_ci_lower", "ci_upper": "balanced_accuracy_ci_upper",
    })
    worst = subgroups[
        subgroups["grouping_variable"].eq("batch_chromatography")
    ].groupby("model")["balanced_accuracy"].min().rename("worst_batch_balanced_accuracy")
    model_summary = model_summary.merge(ba_intervals, on="model", how="left")
    model_summary = model_summary.drop(columns=["averaged_seed_balanced_accuracy"]).merge(
        averaged_metrics, on="model", how="left", validate="one_to_one"
    )
    model_summary["worst_batch_balanced_accuracy"] = model_summary["model"].map(worst)
    model_summary = model_summary.sort_values("averaged_seed_balanced_accuracy", ascending=False)
    return {
        "averaged_predictions": averaged,
        "seed_metrics": seed_metrics,
        "subgroup_metrics": subgroups,
        "bootstrap_intervals": interval_frame,
        "bootstrap_replicates": pd.concat(bootstrap_frames, ignore_index=True),
        "paired_comparisons": pd.DataFrame(paired_rows),
        "paired_replicates": pd.concat(paired_frames, ignore_index=True),
        "calibration": pd.concat(reliability_frames, ignore_index=True),
        "model_summary": model_summary,
    }


def original_comparison(model_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sources = [
        (Path("outputs/tables/stage_06_model_summary.csv"), "model", "stage_06"),
        (Path("outputs/tables/stage_08_model_summary.csv"), "model", "stage_08"),
        (Path("outputs/tables/stage_12_model_summary.csv"), "experiment", "stage_12"),
    ]
    for path, model_column, stage in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            model = str(getattr(row, model_column))
            if model in MODEL_LABELS:
                rows.append({
                    "model": model, "model_label": MODEL_LABELS[model], "corpus": "proof_of_concept",
                    "source_stage": stage, "cells": 215 if stage != "stage_12" else 202,
                    "balanced_accuracy": float(getattr(row, "balanced_accuracy_mean")),
                    "macro_f1": float(getattr(row, "macro_f1_mean")),
                    "auroc": float(getattr(row, "auroc_mean")),
                })
    expanded = model_summary[[
        "model", "model_label", "averaged_seed_balanced_accuracy",
        "averaged_seed_macro_f1", "averaged_seed_auroc",
    ]].copy()
    expanded = expanded.rename(columns={
        "averaged_seed_balanced_accuracy": "balanced_accuracy",
        "averaged_seed_macro_f1": "macro_f1", "averaged_seed_auroc": "auroc",
    })
    expanded["corpus"] = "expanded_stage_15"
    expanded["source_stage"] = "stage_15"
    expanded["cells"] = 1046
    return pd.concat([pd.DataFrame(rows), expanded], ignore_index=True)


def make_figures(
    model_summary: pd.DataFrame, calibration: pd.DataFrame,
    comparison: pd.DataFrame, runtime: pd.DataFrame, figure_directory: Path,
) -> None:
    ordered = model_summary.sort_values("averaged_seed_balanced_accuracy")
    figure, axis = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    x = ordered["averaged_seed_balanced_accuracy"].to_numpy(float)
    lower = x - ordered["balanced_accuracy_ci_lower"].to_numpy(float)
    upper = ordered["balanced_accuracy_ci_upper"].to_numpy(float) - x
    axis.errorbar(x, ordered["model_label"], xerr=np.vstack([lower, upper]), fmt="o", capsize=3)
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Out-of-fold balanced accuracy with 95% group-bootstrap CI")
    axis.set_title("Stage 15 expanded-corpus model comparison")
    figure.savefig(figure_directory / "stage_15_model_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for model, frame in calibration.groupby("model"):
        occupied = frame[frame["sample_count"].gt(0)]
        axis.plot(occupied["mean_probability"], occupied["observed_positive_fraction"], marker="o", label=MODEL_LABELS[model])
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed sc_u fraction", title="Stage 15 reliability curves")
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(figure_directory / "stage_15_calibration.png", dpi=180)
    plt.close(figure)

    paired = comparison.pivot_table(index=["model", "model_label"], columns="corpus", values="balanced_accuracy", aggfunc="mean").dropna()
    if not paired.empty:
        figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        positions = np.arange(len(paired))
        width = 0.36
        axis.bar(positions - width / 2, paired["proof_of_concept"], width, label="Proof of concept")
        axis.bar(positions + width / 2, paired["expanded_stage_15"], width, label="Expanded Stage 15")
        axis.set_xticks(positions, paired.index.get_level_values("model_label"), rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Balanced accuracy")
        axis.set_title("Original and expanded-corpus results")
        axis.legend()
        figure.savefig(figure_directory / "stage_15_original_comparison.png", dpi=180)
        plt.close(figure)

    totals = runtime.groupby(["model", "model_family"], as_index=False)["runtime_seconds"].sum().sort_values("runtime_seconds")
    figure, axis = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    axis.barh(totals["model"].map(MODEL_LABELS), totals["runtime_seconds"] / 60)
    axis.set_xlabel("Total training and evaluation runtime (minutes)")
    axis.set_title("Stage 15 measured CPU runtime")
    figure.savefig(figure_directory / "stage_15_runtime.png", dpi=180)
    plt.close(figure)


def render_report(summary: dict[str, Any], model_summary: pd.DataFrame) -> str:
    result_lines = [
        f"- `{row.model_label}`: balanced accuracy {row.averaged_seed_balanced_accuracy:.3f} "
        f"(95% group-bootstrap CI {row.balanced_accuracy_ci_lower:.3f}–{row.balanced_accuracy_ci_upper:.3f}); "
        f"macro F1 {row.averaged_seed_macro_f1:.3f}; AUROC {row.averaged_seed_auroc:.3f}."
        for row in model_summary.itertuples(index=False)
    ]
    return "\n".join([
        "# Stage 15 — Expanded-corpus retraining", "",
        "## Corpus and design", "",
        f"- Matched cells: {summary['cells']:,} (`sc_m0` = {summary['negative_cells']:,}; `sc_u` = {summary['positive_cells']:,}).",
        f"- Raw-file groups: {summary['raw_file_groups']:,}; chromatography batches: {', '.join(summary['batches'])}.",
        "- All models use the same five raw-file-grouped outer folds.",
        "- Conventional hyperparameters use all three saved grouped inner folds. Neural epoch selection uses the prespecified grouped inner fold 0, identically for every neural model and seed.",
        "- Prevalence filtering, imputation, scaling, class weights, and early stopping are fitted using training data only.",
        "- The decision threshold is prespecified at 0.5; no outer-test threshold tuning or calibration fitting is performed.",
        f"- Neural models use deterministic seeds {', '.join(map(str, summary['neural_seeds']))}.", "",
        "## Results", "", *result_lines, "",
        f"- Highest expanded-corpus balanced accuracy: `{summary['best_model_label']}` = {summary['best_balanced_accuracy']:.3f}.",
        "- Batch-only performance is a diagnostic for residual technical confounding, not a biological model.",
        "- Neural seed probabilities are averaged only for the primary model-level comparison; every seed-specific OOF prediction remains available.", "",
        "## Uncertainty and robustness", "",
        f"- Confidence intervals use {summary['bootstrap_replicates']:,} paired or unpaired raw-file-group bootstrap replicates.",
        "- Subgroup metrics include chromatography batch and source cohort, with unsupported one-class or very small groups left blank.",
        "- Calibration tables report uncalibrated Brier score, expected calibration error, and reliability bins.", "",
        "## Outputs and stage gate", "",
        f"- Saved {summary['prediction_rows']:,} seed-specific OOF prediction rows and {summary['checkpoint_files']:,} model checkpoints.",
        f"- Total measured runtime: {summary['runtime_minutes']:.1f} minutes on `{summary['device']}`; peak process memory {summary['peak_memory_mib']:.1f} MiB.",
        "- Stage 15 passes because every aligned cell has exactly one OOF prediction per model and applicable seed, all raw-file grouping checks pass, and performance is reported for every requested model.", "",
    ])


def run_stage_fifteen(
    config_path: str | Path = "configs/stage_15.yaml", dry_run: bool = False,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    data = config["data"]
    directory = Path(data["directory"])
    manifest = pd.read_parquet(data["manifest"])
    peptide = pd.read_parquet(data["peptide_matrix"])
    protein = pd.read_parquet(data["protein_matrix"])
    spectrum_bags = pd.read_parquet(directory / "spectrum_bag_index.parquet")
    spectrum_metadata = pd.read_parquet(directory / "spectrum_metadata.parquet")
    spectra_path = directory / "spectra.h5"
    reporter_path = directory / "reporter_intensities.h5"
    outer = pd.read_csv(data["outer_splits"])
    inner = pd.read_csv(data["inner_splits"])
    validate_inputs(manifest, peptide, protein, spectrum_bags, outer, inner)
    validate_spectrum_bags(spectrum_bags, spectrum_metadata, spectra_path, reporter_path)
    vocabulary, peptide_instances, peptide_bags = make_peptide_bags(config)
    if set(peptide_bags["sample_id"]) != set(manifest["sample_id"]):
        raise ValueError("peptide bags do not cover the complete Stage 14 corpus")
    if dry_run:
        return {
            "dry_run": True, "cells": len(manifest), "peptide_features": len(peptide),
            "protein_features": len(protein), "peptide_instances": len(peptide_instances),
            "spectrum_bags": len(spectrum_bags), "models": len(MODEL_LABELS),
        }
    checkpoint_directory = Path("outputs/checkpoints/stage_15")
    table_directory = Path("outputs/tables")
    figure_directory = Path("outputs/figures")
    report_directory = Path("outputs/reports")
    for path in (checkpoint_directory, table_directory, figure_directory, report_directory):
        path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config["runtime"]["device"]))
    started = time.perf_counter()
    progress = load_progress(table_directory, bool(config["runtime"].get("resume", True)))
    train_conventional(
        config, manifest, peptide, protein, outer, inner,
        checkpoint_directory, table_directory, progress,
    )
    train_peptide_mil(
        config, vocabulary, peptide_instances, peptide_bags, outer, inner,
        checkpoint_directory, table_directory, progress, device,
    )
    train_spectrum_mil(
        config, spectrum_bags, spectrum_metadata, spectra_path, reporter_path, outer, inner,
        checkpoint_directory, table_directory, progress, device,
    )
    predictions = pd.DataFrame(progress["predictions"])
    fold_metrics = pd.DataFrame(progress["fold_metrics"])
    history = pd.DataFrame(progress["history"])
    runtime = pd.DataFrame(progress["runtime"])
    tuning = pd.DataFrame(progress["tuning"])
    validate_predictions(
        predictions, set(manifest["sample_id"].astype(str)), int(config["seed"]),
        [int(value) for value in config["neural_seeds"]],
    )
    evaluated = evaluate_models(config, predictions, manifest)
    comparison = original_comparison(evaluated["model_summary"])
    split_audit = pd.concat([
        outer.groupby(["outer_fold", "partition", "label_raw"]).agg(
            cells=("sample_id", "size"), raw_file_groups=("raw_file_stem", "nunique")
        ).reset_index().assign(split_level="outer", inner_fold=np.nan),
        inner.groupby(["outer_fold", "inner_fold", "partition", "label_raw"]).agg(
            cells=("sample_id", "size"), raw_file_groups=("raw_file_stem", "nunique")
        ).reset_index().assign(split_level="inner"),
    ], ignore_index=True, sort=False)
    predictions.to_csv(table_directory / "stage_15_predictions.csv", index=False)
    fold_metrics.to_csv(table_directory / "stage_15_fold_seed_metrics.csv", index=False)
    evaluated["seed_metrics"].to_csv(table_directory / "stage_15_overall_seed_metrics.csv", index=False)
    evaluated["model_summary"].to_csv(table_directory / "stage_15_model_summary.csv", index=False)
    evaluated["subgroup_metrics"].to_csv(table_directory / "stage_15_subgroup_metrics.csv", index=False)
    evaluated["calibration"].to_csv(table_directory / "stage_15_calibration.csv", index=False)
    evaluated["bootstrap_intervals"].to_csv(table_directory / "stage_15_bootstrap_intervals.csv", index=False)
    evaluated["paired_comparisons"].to_csv(table_directory / "stage_15_paired_comparisons.csv", index=False)
    history.to_csv(table_directory / "stage_15_training_history.csv", index=False)
    runtime.to_csv(table_directory / "stage_15_runtime_memory.csv", index=False)
    tuning.to_csv(table_directory / "stage_15_tuning.csv", index=False)
    comparison.to_csv(table_directory / "stage_15_original_comparison.csv", index=False)
    split_audit.to_csv(table_directory / "stage_15_split_audit.csv", index=False)
    evaluated["averaged_predictions"].to_parquet(
        table_directory / "stage_15_averaged_seed_predictions.parquet", index=False
    )
    evaluated["bootstrap_replicates"].to_parquet(
        table_directory / "stage_15_bootstrap_replicates.parquet", index=False
    )
    evaluated["paired_replicates"].to_parquet(
        table_directory / "stage_15_paired_bootstrap_replicates.parquet", index=False
    )
    make_figures(
        evaluated["model_summary"], evaluated["calibration"], comparison, runtime,
        figure_directory,
    )
    best = evaluated["model_summary"].iloc[0]
    elapsed = time.perf_counter() - started
    label_counts = manifest["label_raw"].value_counts()
    summary = {
        "cells": int(len(manifest)), "negative_cells": int(label_counts.get("sc_m0", 0)),
        "positive_cells": int(label_counts.get("sc_u", 0)),
        "raw_file_groups": int(manifest["raw_file_stem"].nunique()),
        "batches": sorted(manifest["batch_chromatography"].astype(str).unique()),
        "neural_seeds": [int(value) for value in config["neural_seeds"]],
        "models": len(MODEL_LABELS), "prediction_rows": int(len(predictions)),
        "checkpoint_files": len(list(checkpoint_directory.glob("*"))),
        "bootstrap_replicates": int(config["evaluation"]["bootstrap_replicates"]),
        "best_model": str(best["model"]), "best_model_label": str(best["model_label"]),
        "best_balanced_accuracy": float(best["averaged_seed_balanced_accuracy"]),
        "runtime_seconds": elapsed, "runtime_minutes": elapsed / 60,
        "peak_memory_mib": resident_memory_mib(), "device": str(device),
        "peptide_features": int(peptide.shape[0]), "protein_features": int(protein.shape[0]),
        "peptide_instances": int(len(peptide_instances)),
        "stored_spectra": int(len(spectrum_metadata)),
    }
    (report_directory / "stage_15_expanded_corpus_models.md").write_text(
        render_report(summary, evaluated["model_summary"]), encoding="utf-8"
    )
    (table_directory / "stage_15_run_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for name in PROGRESS_NAMES:
        path = table_directory / f"stage_15_{name}.partial.csv"
        if path.exists():
            path.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage_15.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_15_expanded_corpus_models.log"))
    result = run_stage_fifteen(args.config, args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 15 completed: %s", result)


if __name__ == "__main__":
    main()
