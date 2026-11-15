"""Train and evaluate Stage 12 spectrum-aware MIL on grouped outer folds."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import binary_metrics
from src.models.spectrum_mil import SpectrumAttentionMIL
from src.training.spectrum_datasets import (
    BASE_NUMERIC_DIM,
    PSM_NUMERIC_DIM,
    SpectrumArrayStore,
    SpectrumBagDataset,
    VectorStandardizer,
    collate_spectrum_bags,
    fit_numeric_standardizer,
    make_psm_vocabulary,
    validate_spectrum_bags,
)
from src.training.train_mil import resolve_device
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging
from src.utils.seeds import seed_everything


VARIANT_LABELS = {
    "reporter_only": "Reporter + precursor",
    "spectrum_embedding": "Spectrum + reporter",
    "spectrum_psm": "Spectrum + reporter + PSM",
}
PROCESSED_MODEL_LABELS = {
    "peptide_logistic": "Processed peptide logistic",
    "max_pooling": "Processed max-pooling MIL",
    "attention": "Processed attention MIL",
}
METRIC_COLUMNS = [
    "balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc",
    "sensitivity", "specificity", "accuracy",
]


def variant_options(variant: str) -> tuple[bool, bool, int]:
    """Return use-spectrum, use-PSM, and numeric dimension for a named variant."""
    if variant == "reporter_only":
        return False, False, BASE_NUMERIC_DIM
    if variant == "spectrum_embedding":
        return True, False, BASE_NUMERIC_DIM
    if variant == "spectrum_psm":
        return True, True, BASE_NUMERIC_DIM + PSM_NUMERIC_DIM
    raise ValueError(f"unknown spectrum MIL variant: {variant}")


def validate_grouped_subset_splits(
    bags: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    validation_inner_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subset immutable Stage 5 folds while preserving all grouping guarantees."""
    sample_ids = set(bags["sample_id"])
    outer_subset = outer[outer["sample_id"].isin(sample_ids)].copy()
    inner_subset = inner[inner["sample_id"].isin(sample_ids)].copy()
    if set(outer_subset["sample_id"]) != sample_ids:
        raise ValueError("outer folds do not cover all spectrum bags")
    test_counts = outer_subset.loc[
        outer_subset["partition"].eq("test"), "sample_id"
    ].value_counts()
    if set(test_counts.index) != sample_ids or not test_counts.eq(1).all():
        raise ValueError("each spectrum cell must occur in exactly one outer test fold")
    for outer_fold, frame in outer_subset.groupby("outer_fold"):
        if frame.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
            raise ValueError(f"raw file crosses outer fold {outer_fold}")
        if frame.groupby("partition")["label_raw"].nunique().lt(2).any():
            raise ValueError(f"outer fold {outer_fold} lacks both labels")
        training_ids = set(frame.loc[frame["partition"].eq("train"), "sample_id"])
        selected = inner_subset[
            inner_subset["outer_fold"].eq(outer_fold)
            & inner_subset["inner_fold"].eq(validation_inner_fold)
        ]
        if set(selected["sample_id"]) != training_ids:
            raise ValueError(f"inner fold does not cover outer-training cells: {outer_fold}")
        if selected.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
            raise ValueError(f"raw file crosses inner split {outer_fold}")
        if selected.groupby("partition")["label_raw"].nunique().lt(2).any():
            raise ValueError(f"inner split lacks both labels: {outer_fold}")
    return outer_subset, inner_subset


def build_spectrum_model(
    variant: str,
    source_spectrum_dim: int,
    vocabulary_size: int,
    model_config: dict[str, Any],
) -> SpectrumAttentionMIL:
    use_spectrum, use_psm, numeric_dim = variant_options(variant)
    factor = int(model_config["spectrum_downsample_factor"])
    if source_spectrum_dim % factor != 0:
        raise ValueError("source spectrum dimension is not divisible by configured factor")
    return SpectrumAttentionMIL(
        spectrum_feature_dim=source_spectrum_dim // factor if use_spectrum else 0,
        numeric_feature_dim=numeric_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        spectrum_projection_dim=int(model_config["spectrum_projection_dim"]),
        peptide_vocab_size=vocabulary_size if use_psm else 0,
        peptide_embedding_dim=int(model_config["peptide_embedding_dim"]),
        dropout=float(model_config["dropout"]),
    )


def make_loader(
    dataset: SpectrumBagDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_spectrum_bags,
        generator=torch.Generator().manual_seed(seed),
    )


def positive_class_weight(
    bags: pd.DataFrame, sample_ids: list[str], device: torch.device
) -> torch.Tensor:
    labels = bags.set_index("sample_id").loc[sample_ids, "label_binary"].to_numpy(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training cells must contain both labels")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def _forward(
    model: SpectrumAttentionMIL,
    batch: dict[str, Any],
    device: torch.device,
    return_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return model(
        batch["spectrum_features"].to(device),
        batch["numeric_features"].to(device),
        batch["peptide_ids"].to(device),
        batch["bag_index"].to(device),
        return_attention=return_attention,
    )


def train_epoch(
    model: SpectrumAttentionMIL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
    accumulation_steps: int,
) -> float:
    """Train one epoch with optional gradient accumulation."""
    if accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, bag_count = 0.0, 0
    for batch_index, batch in enumerate(loader):
        labels = batch["labels"].to(device)
        logits = _forward(model, batch, device)
        loss = criterion(logits, labels)
        (loss / accumulation_steps).backward()
        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach()) * len(labels)
        bag_count += len(labels)
    return loss_sum / bag_count


def evaluate_once(
    model: SpectrumAttentionMIL,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_attention: bool = False,
) -> dict[str, Any]:
    """Evaluate one deterministic bag realization."""
    model.eval()
    labels_all, probabilities_all = [], []
    sample_ids_all: list[str] = []
    attention_frames = []
    loss_sum, bag_count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            if collect_attention:
                logits, weights = _forward(model, batch, device, True)
            else:
                logits = _forward(model, batch, device)
                weights = None
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())
            sample_ids_all.extend(batch["sample_ids"])
            loss_sum += float(loss) * len(labels)
            bag_count += len(labels)
            if weights is not None:
                local_bag = batch["bag_index"].numpy()
                local_samples = np.asarray(batch["sample_ids"], dtype=object)
                attention_frames.append(pd.DataFrame({
                    "sample_id": local_samples[local_bag],
                    "spectrum_id": batch["spectrum_ids"].numpy(),
                    "reporter_intensity": batch["reporter_intensity"].numpy(),
                    "attention_weight": weights.cpu().numpy(),
                }))
    y_true = np.concatenate(labels_all).astype(int)
    probabilities = np.concatenate(probabilities_all)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "loss": loss_sum / bag_count,
        "sample_ids": sample_ids_all,
        "y_true": y_true,
        "y_pred": predictions,
        "probabilities": probabilities,
        "metrics": binary_metrics(y_true, predictions, probabilities),
        "attention": pd.concat(attention_frames, ignore_index=True)
        if attention_frames else pd.DataFrame(),
    }


def evaluate_repeated(
    model: SpectrumAttentionMIL,
    dataset: SpectrumBagDataset,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    seed: int,
    repeats: int,
    collect_attention: bool,
) -> dict[str, Any]:
    """Average probabilities over deterministic instance subsamples."""
    results = []
    for repeat in range(repeats):
        dataset.set_epoch(10_000 + repeat)
        loader = make_loader(dataset, batch_size, False, seed + repeat)
        results.append(evaluate_once(
            model, loader, criterion, device,
            collect_attention=collect_attention and repeat == 0,
        ))
    canonical = results[0]
    for result in results[1:]:
        if result["sample_ids"] != canonical["sample_ids"] or not np.array_equal(
            result["y_true"], canonical["y_true"]
        ):
            raise AssertionError("repeated evaluation samples are misaligned")
    probabilities = np.mean([result["probabilities"] for result in results], axis=0)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "loss": float(np.mean([result["loss"] for result in results])),
        "sample_ids": canonical["sample_ids"],
        "y_true": canonical["y_true"],
        "y_pred": predictions,
        "probabilities": probabilities,
        "metrics": binary_metrics(canonical["y_true"], predictions, probabilities),
        "attention": canonical["attention"],
    }


def select_epoch_count(
    experiment_name: str,
    model: SpectrumAttentionMIL,
    training_dataset: SpectrumBagDataset,
    validation_dataset: SpectrumBagDataset,
    bags: pd.DataFrame,
    training_ids: list[str],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Select training duration using one untouched grouped inner validation split."""
    train_loader = make_loader(
        training_dataset, int(config["batch_size"]), True, training_dataset.seed
    )
    validation_loader = make_loader(
        validation_dataset, int(config["evaluation_batch_size"]), False,
        validation_dataset.seed,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weight(bags, training_ids, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_score, best_loss, best_epoch = -np.inf, np.inf, 0
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history = []
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        training_dataset.set_epoch(epoch)
        training_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            float(config["gradient_clip_norm"]),
            int(config["gradient_accumulation_steps"]),
        )
        validation_dataset.set_epoch(0)
        validation = evaluate_once(model, validation_loader, criterion, device)
        score = float(validation["metrics"]["balanced_accuracy"])
        validation_loss = float(validation["loss"])
        history.append({
            "experiment": experiment_name,
            "phase": "selection",
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in validation["metrics"].items()},
        })
        improved = score > best_score + float(config["early_stopping_min_delta"]) or (
            abs(score - best_score) <= float(config["early_stopping_min_delta"])
            and validation_loss < best_loss
        )
        if improved:
            best_score, best_loss, best_epoch = score, validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch >= int(config["minimum_epochs"]) and without_improvement >= int(
            config["early_stopping_patience"]
        ):
            break
    if best_state is None:
        raise RuntimeError("early stopping failed to select a spectrum checkpoint")
    return best_epoch, best_score, history, best_state


def fit_final_model(
    experiment_name: str,
    model: SpectrumAttentionMIL,
    dataset: SpectrumBagDataset,
    bags: pd.DataFrame,
    training_ids: list[str],
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    loader = make_loader(dataset, int(config["batch_size"]), True, dataset.seed)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weight(bags, training_ids, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        loss = train_epoch(
            model, loader, optimizer, criterion, device,
            float(config["gradient_clip_norm"]),
            int(config["gradient_accumulation_steps"]),
        )
        history.append({
            "experiment": experiment_name, "phase": "final_fit",
            "epoch": epoch, "training_loss": loss,
        })
    return history


def checkpoint_payload(
    stage_variant: str,
    experiment: str,
    model: SpectrumAttentionMIL,
    outer_fold: int,
    seed: int,
    epochs: int,
    standardizer: VectorStandardizer,
    training_ids: list[str],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": 12,
        "variant": stage_variant,
        "experiment": experiment,
        "outer_fold": outer_fold,
        "seed": seed,
        "epochs": epochs,
        "numeric_standardizer": standardizer.as_dict(),
        "training_sample_ids": training_ids,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }


def peak_memory_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby("experiment")[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index().sort_values("balanced_accuracy_mean", ascending=False)


def validate_stage_twelve_outputs(
    predictions: pd.DataFrame,
    attention: pd.DataFrame,
    bags: pd.DataFrame,
    variants: list[str],
) -> None:
    """Enforce Stage 12 OOF-prediction and attention integrity guarantees."""
    expected_ids = set(bags["sample_id"].astype(str))
    expected_labels = bags.set_index("sample_id")["label_binary"].astype(int)
    primary = predictions[predictions["analysis_type"].eq("primary_variant")]
    for variant in variants:
        frame = primary[primary["experiment"].eq(variant)]
        if len(frame) != len(expected_ids) or set(frame["sample_id"]) != expected_ids:
            raise ValueError(f"{variant} does not contain one prediction for every cell")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{variant} contains duplicate OOF cell predictions")
        if not np.array_equal(
            frame["y_true"].to_numpy(dtype=int),
            expected_labels.loc[frame["sample_id"]].to_numpy(dtype=int),
        ):
            raise ValueError(f"{variant} prediction labels disagree with the bag index")
        probabilities = frame["probability_sc_u"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or not (
            (probabilities >= 0.0) & (probabilities <= 1.0)
        ).all():
            raise ValueError(f"{variant} contains invalid probabilities")

    if set(attention["sample_id"]) != expected_ids:
        raise ValueError("attention output does not cover every spectrum cell")
    if attention.duplicated(["sample_id", "spectrum_id"]).any():
        raise ValueError("attention output contains duplicate spectra within a cell")
    weights = attention["attention_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("attention output contains invalid weights")
    weight_sums = attention.groupby("sample_id")["attention_weight"].sum().to_numpy()
    if not np.allclose(weight_sums, 1.0, atol=1e-5):
        raise ValueError("attention weights do not sum to one within every sampled bag")


def processed_comparison(
    prediction_paths: list[str | Path],
    raw_predictions: pd.DataFrame,
    sample_ids: set[str],
    primary_variants: list[str],
) -> pd.DataFrame:
    """Compare OOF performance on the same 202 test cells."""
    frames = []
    for path in prediction_paths:
        frame = pd.read_csv(path)
        frame = frame[frame["sample_id"].isin(sample_ids)]
        frames.append(frame)
    processed = pd.concat(frames, ignore_index=True)
    records = []
    for model_name in ["peptide_logistic", "max_pooling", "attention"]:
        frame = processed[processed["model"].eq(model_name)]
        if len(frame) != len(sample_ids):
            raise ValueError(f"processed comparison model lacks 202-cell coverage: {model_name}")
        records.append({
            "model": f"processed_{model_name}",
            "model_label": PROCESSED_MODEL_LABELS[model_name],
            "source": "processed_data",
            "training_corpus_note": "trained on original 215-cell corpus; evaluated on 202 common cells",
            **binary_metrics(
                frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int),
                frame["probability_sc_u"].to_numpy(float),
            ),
        })
    for variant in primary_variants:
        frame = raw_predictions[raw_predictions["experiment"].eq(variant)]
        if len(frame) != len(sample_ids):
            raise ValueError(f"raw-spectrum variant lacks OOF coverage: {variant}")
        records.append({
            "model": variant,
            "model_label": VARIANT_LABELS[variant],
            "source": "raw_spectrum",
            "training_corpus_note": "trained and evaluated on 202-cell raw-spectrum corpus",
            **binary_metrics(
                frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int),
                frame["probability_sc_u"].to_numpy(float),
            ),
        })
    return pd.DataFrame(records).sort_values("balanced_accuracy", ascending=False)


def comparison_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    metrics = ["balanced_accuracy", "macro_f1", "auroc"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    order = comparison["model"].tolist()
    labels = comparison.set_index("model")["model_label"]
    colors = ["tab:blue" if comparison.set_index("model").loc[name, "source"] == "raw_spectrum" else "tab:gray" for name in order]
    for axis, metric in zip(axes, metrics):
        values = comparison.set_index("model").loc[order, metric]
        axis.bar(np.arange(len(order)), values, color=colors)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_xticks(range(len(order)), [labels[name] for name in order], rotation=35, ha="right")
        axis.set_title(metric.replace("_", " ").title())
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def learning_curve_plot(history: pd.DataFrame, output_path: Path) -> None:
    selection = history[
        history["phase"].eq("selection") & history["experiment"].isin(VARIANT_LABELS)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True, constrained_layout=True)
    for axis, variant in zip(axes, VARIANT_LABELS):
        selected = selection[selection["experiment"].eq(variant)]
        for fold, frame in selected.groupby("outer_fold"):
            axis.plot(frame["epoch"], frame["validation_balanced_accuracy"], label=f"Fold {fold}")
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_title(VARIANT_LABELS[variant])
        axis.set_xlabel("Epoch")
    axes[0].set_ylabel("Inner-validation balanced accuracy")
    axes[-1].legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def bag_size_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    selected = metrics[metrics["analysis_type"].eq("bag_size_sensitivity")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, metric in zip(axes, ["balanced_accuracy", "auroc"]):
        for size, frame in selected.groupby("maximum_training_instances"):
            axis.scatter(np.full(len(frame), size), frame[metric], alpha=0.65)
        means = selected.groupby("maximum_training_instances")[metric].mean().sort_index()
        axis.plot(means.index, means.values, marker="o", linewidth=2)
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(means.index, means.index.astype(int))
        axis.set_ylim(0, 1)
        axis.set_xlabel("Maximum spectra per bag")
        axis.set_title(metric.replace("_", " ").title())
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_stage_twelve(
    config_path: str | Path = "configs/spectrum_mil.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run nested grouped training for spectrum-aware MIL variants."""
    root_config = load_yaml_config(config_path)
    config = root_config["stage_12"]
    training = config["training"]
    model_config = config["model"]
    data_directory = Path(config["data_directory"])
    bags = pd.read_parquet(data_directory / "cell_bag_index.parquet")
    metadata = pd.read_parquet(data_directory / "spectrum_metadata.parquet")
    spectra_path = data_directory / "spectra.h5"
    reporter_path = data_directory / "reporter_intensities.h5"
    validate_spectrum_bags(bags, metadata, spectra_path, reporter_path)
    outer = pd.read_csv(config["outer_splits"])
    inner = pd.read_csv(config["inner_splits"])
    validation_inner_fold = int(training["validation_inner_fold"])
    outer, inner = validate_grouped_subset_splits(
        bags, outer, inner, validation_inner_fold
    )
    variants = [str(value) for value in config["variants"]]
    for variant in variants:
        variant_options(variant)
    primary_variant = str(config["primary_variant"])
    if primary_variant not in variants:
        raise ValueError("primary spectrum variant must be among configured variants")
    peptide_to_id, vocabulary = make_psm_vocabulary(metadata)
    vocabulary_size = len(vocabulary) + 1
    import h5py
    with h5py.File(spectra_path, "r") as handle:
        source_spectrum_dim = int(handle["spectrum_embedding"].shape[1])
    array_store = SpectrumArrayStore.from_hdf5(
        spectra_path,
        reporter_path,
        int(model_config["spectrum_downsample_factor"]),
    )
    device = resolve_device(str(training["device"]))
    seed = int(root_config["seed"])
    seed_everything(seed)
    if dry_run:
        fold = int(outer["outer_fold"].min())
        frame = outer[outer["outer_fold"].eq(fold)]
        training_ids = frame.loc[frame["partition"].eq("train"), "sample_id"].tolist()
        output_shapes = {}
        for variant in variants:
            use_spectrum, use_psm, _ = variant_options(variant)
            standardizer = fit_numeric_standardizer(
                bags, metadata, training_ids[:8], spectra_path, reporter_path, use_psm,
                array_store,
            )
            dataset = SpectrumBagDataset(
                bags, metadata, training_ids[:2], spectra_path, reporter_path,
                standardizer, peptide_to_id, use_spectrum, use_psm,
                int(model_config["spectrum_downsample_factor"]), 32, seed, array_store,
            )
            batch = collate_spectrum_bags([dataset[0], dataset[1]])
            model = build_spectrum_model(
                variant, source_spectrum_dim, vocabulary_size, model_config
            ).to(device).eval()
            with torch.no_grad():
                logits, attention = _forward(model, batch, device, True)
            output_shapes[variant] = {
                "logits": list(logits.shape),
                "attention": list(attention.shape),
                "numeric_features": list(batch["numeric_features"].shape),
                "spectrum_features": list(batch["spectrum_features"].shape),
            }
            dataset.close()
        return {
            "dry_run": True,
            "cells": int(len(bags)),
            "raw_file_groups": int(bags["raw_file_stem"].nunique()),
            "cached_spectra": int(len(metadata)),
            "source_spectrum_dimensions": source_spectrum_dim,
            "psm_vocabulary_size_including_unknown": vocabulary_size,
            "variant_forward_shapes": output_shapes,
        }
    checkpoint_directory = Path("outputs/checkpoints/stage_12")
    table_directory = Path("outputs/tables")
    report_directory = Path("outputs/reports")
    figure_directory = Path("outputs/figures")
    for directory in (checkpoint_directory, table_directory, report_directory, figure_directory):
        directory.mkdir(parents=True, exist_ok=True)
    vocabulary.to_parquet(data_directory / "psm_peptide_vocabulary.parquet", index=False)
    label_by_sample = bags.set_index("sample_id")["label_raw"]
    all_predictions, fold_metrics, history, attention_frames, runtime_records = [], [], [], [], []
    selected_epochs: dict[int, int] = {}
    started = time.perf_counter()
    logger = logging.getLogger("single_cell_proteomics_mil")
    for outer_fold in sorted(outer["outer_fold"].unique()):
        outer_frame = outer[outer["outer_fold"].eq(outer_fold)]
        outer_training_ids = outer_frame.loc[
            outer_frame["partition"].eq("train"), "sample_id"
        ].tolist()
        test_ids = outer_frame.loc[outer_frame["partition"].eq("test"), "sample_id"].tolist()
        inner_frame = inner[
            inner["outer_fold"].eq(outer_fold)
            & inner["inner_fold"].eq(validation_inner_fold)
        ]
        selection_ids = inner_frame.loc[
            inner_frame["partition"].eq("train"), "sample_id"
        ].tolist()
        validation_ids = inner_frame.loc[
            inner_frame["partition"].eq("validation"), "sample_id"
        ].tolist()
        for variant_index, variant in enumerate(variants):
            experiment_start = time.perf_counter()
            experiment_seed = seed + int(outer_fold) * 1_000 + variant_index * 10_000
            use_spectrum, use_psm, _ = variant_options(variant)
            logger.info("Stage 12 | fold=%s variant=%s | selecting epochs", outer_fold, variant)
            seed_everything(experiment_seed)
            selection_standardizer = fit_numeric_standardizer(
                bags, metadata, selection_ids, spectra_path, reporter_path, use_psm,
                array_store,
            )
            selection_dataset = SpectrumBagDataset(
                bags, metadata, selection_ids, spectra_path, reporter_path,
                selection_standardizer, peptide_to_id, use_spectrum, use_psm,
                int(model_config["spectrum_downsample_factor"]),
                int(training["maximum_training_instances_per_bag"]), experiment_seed,
                array_store,
            )
            validation_dataset = SpectrumBagDataset(
                bags, metadata, validation_ids, spectra_path, reporter_path,
                selection_standardizer, peptide_to_id, use_spectrum, use_psm,
                int(model_config["spectrum_downsample_factor"]),
                int(training["maximum_evaluation_instances_per_bag"]), experiment_seed,
                array_store,
            )
            selection_model = build_spectrum_model(
                variant, source_spectrum_dim, vocabulary_size, model_config
            ).to(device)
            best_epoch, best_score, selected_history, best_state = select_epoch_count(
                variant, selection_model, selection_dataset, validation_dataset,
                bags, selection_ids, training, device,
            )
            for record in selected_history:
                record["outer_fold"] = int(outer_fold)
                record["variant"] = variant
            history.extend(selected_history)
            torch.save(
                checkpoint_payload(
                    variant, variant, selection_model, int(outer_fold), experiment_seed,
                    best_epoch, selection_standardizer, selection_ids, model_config,
                ) | {
                    "validation_sample_ids": validation_ids,
                    "best_validation_balanced_accuracy": best_score,
                    "model_state_dict": best_state,
                },
                checkpoint_directory / f"{variant}_fold_{outer_fold}_selection.pt",
            )
            selection_dataset.close()
            validation_dataset.close()
            if variant == primary_variant:
                selected_epochs[int(outer_fold)] = best_epoch
            seed_everything(experiment_seed)
            final_standardizer = fit_numeric_standardizer(
                bags, metadata, outer_training_ids, spectra_path, reporter_path, use_psm,
                array_store,
            )
            final_dataset = SpectrumBagDataset(
                bags, metadata, outer_training_ids, spectra_path, reporter_path,
                final_standardizer, peptide_to_id, use_spectrum, use_psm,
                int(model_config["spectrum_downsample_factor"]),
                int(training["maximum_training_instances_per_bag"]), experiment_seed,
                array_store,
            )
            final_model = build_spectrum_model(
                variant, source_spectrum_dim, vocabulary_size, model_config
            ).to(device)
            final_history = fit_final_model(
                variant, final_model, final_dataset, bags, outer_training_ids,
                best_epoch, training, device,
            )
            for record in final_history:
                record["outer_fold"] = int(outer_fold)
                record["variant"] = variant
            history.extend(final_history)
            final_dataset.close()
            test_dataset = SpectrumBagDataset(
                bags, metadata, test_ids, spectra_path, reporter_path,
                final_standardizer, peptide_to_id, use_spectrum, use_psm,
                int(model_config["spectrum_downsample_factor"]),
                int(training["maximum_evaluation_instances_per_bag"]), experiment_seed,
                array_store,
            )
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=positive_class_weight(bags, outer_training_ids, device)
            )
            test = evaluate_repeated(
                final_model, test_dataset, criterion, device,
                int(training["evaluation_batch_size"]), experiment_seed,
                int(training["evaluation_repeats"]), variant == primary_variant,
            )
            test_dataset.close()
            torch.save(
                checkpoint_payload(
                    variant, variant, final_model, int(outer_fold), experiment_seed,
                    best_epoch, final_standardizer, outer_training_ids, model_config,
                ) | {"test_sample_ids": test_ids},
                checkpoint_directory / f"{variant}_fold_{outer_fold}_final.pt",
            )
            fold_metrics.append({
                "experiment": variant,
                "variant": variant,
                "analysis_type": "primary_variant",
                "outer_fold": int(outer_fold),
                "test_samples": len(test_ids),
                "selected_epochs": best_epoch,
                "best_inner_balanced_accuracy": best_score,
                "maximum_training_instances": int(training["maximum_training_instances_per_bag"]),
                "maximum_evaluation_instances": int(training["maximum_evaluation_instances_per_bag"]),
                **test["metrics"],
            })
            for sample_id, y_true, y_pred, probability in zip(
                test["sample_ids"], test["y_true"], test["y_pred"], test["probabilities"]
            ):
                all_predictions.append({
                    "experiment": variant,
                    "variant": variant,
                    "analysis_type": "primary_variant",
                    "outer_fold": int(outer_fold),
                    "sample_id": sample_id,
                    "label_raw": label_by_sample.loc[sample_id],
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "probability_sc_u": float(probability),
                })
            if variant == primary_variant:
                attention = test["attention"].copy()
                attention["outer_fold"] = int(outer_fold)
                attention_frames.append(attention)
            runtime_records.append({
                "experiment": variant,
                "variant": variant,
                "analysis_type": "primary_variant",
                "outer_fold": int(outer_fold),
                "runtime_seconds": time.perf_counter() - experiment_start,
                "process_peak_memory_mib": peak_memory_mib(),
            })
            logger.info(
                "Stage 12 | fold=%s variant=%s | epoch=%s test_balanced_accuracy=%.3f",
                outer_fold, variant, best_epoch, test["metrics"]["balanced_accuracy"],
            )
    # Evaluate bag-size sensitivity without reusing or inspecting outer-test outcomes.
    sensitivity_sizes = [int(value) for value in training["bag_size_sensitivity"]]
    for size_index, bag_size in enumerate(sensitivity_sizes):
        experiment_name = f"{primary_variant}_bag{bag_size}"
        for outer_fold in sorted(outer["outer_fold"].unique()):
            experiment_start = time.perf_counter()
            frame = outer[outer["outer_fold"].eq(outer_fold)]
            training_ids = frame.loc[frame["partition"].eq("train"), "sample_id"].tolist()
            test_ids = frame.loc[frame["partition"].eq("test"), "sample_id"].tolist()
            experiment_seed = seed + 100_000 + size_index * 10_000 + int(outer_fold) * 1_000
            standardizer = fit_numeric_standardizer(
                bags, metadata, training_ids, spectra_path, reporter_path, False,
                array_store,
            )
            seed_everything(experiment_seed)
            training_dataset = SpectrumBagDataset(
                bags, metadata, training_ids, spectra_path, reporter_path,
                standardizer, peptide_to_id, True, False,
                int(model_config["spectrum_downsample_factor"]), bag_size, experiment_seed,
                array_store,
            )
            model = build_spectrum_model(
                primary_variant, source_spectrum_dim, vocabulary_size, model_config
            ).to(device)
            epochs = selected_epochs[int(outer_fold)]
            size_history = fit_final_model(
                experiment_name, model, training_dataset, bags, training_ids,
                epochs, training, device,
            )
            for record in size_history:
                record["outer_fold"] = int(outer_fold)
                record["variant"] = primary_variant
            history.extend(size_history)
            training_dataset.close()
            test_dataset = SpectrumBagDataset(
                bags, metadata, test_ids, spectra_path, reporter_path,
                standardizer, peptide_to_id, True, False,
                int(model_config["spectrum_downsample_factor"]), bag_size, experiment_seed,
                array_store,
            )
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=positive_class_weight(bags, training_ids, device)
            )
            test = evaluate_repeated(
                model, test_dataset, criterion, device,
                int(training["evaluation_batch_size"]), experiment_seed,
                int(training["bag_size_sensitivity_evaluation_repeats"]), False,
            )
            test_dataset.close()
            torch.save(
                checkpoint_payload(
                    primary_variant, experiment_name, model, int(outer_fold),
                    experiment_seed, epochs, standardizer, training_ids, model_config,
                ) | {"test_sample_ids": test_ids, "maximum_bag_size": bag_size},
                checkpoint_directory / f"{experiment_name}_fold_{outer_fold}_final.pt",
            )
            fold_metrics.append({
                "experiment": experiment_name,
                "variant": primary_variant,
                "analysis_type": "bag_size_sensitivity",
                "outer_fold": int(outer_fold),
                "test_samples": len(test_ids),
                "selected_epochs": epochs,
                "best_inner_balanced_accuracy": np.nan,
                "maximum_training_instances": bag_size,
                "maximum_evaluation_instances": bag_size,
                **test["metrics"],
            })
            for sample_id, y_true, y_pred, probability in zip(
                test["sample_ids"], test["y_true"], test["y_pred"], test["probabilities"]
            ):
                all_predictions.append({
                    "experiment": experiment_name,
                    "variant": primary_variant,
                    "analysis_type": "bag_size_sensitivity",
                    "outer_fold": int(outer_fold),
                    "sample_id": sample_id,
                    "label_raw": label_by_sample.loc[sample_id],
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "probability_sc_u": float(probability),
                })
            runtime_records.append({
                "experiment": experiment_name,
                "variant": primary_variant,
                "analysis_type": "bag_size_sensitivity",
                "outer_fold": int(outer_fold),
                "runtime_seconds": time.perf_counter() - experiment_start,
                "process_peak_memory_mib": peak_memory_mib(),
            })
            logger.info(
                "Stage 12 | fold=%s bag_size=%s | test_balanced_accuracy=%.3f",
                outer_fold, bag_size, test["metrics"]["balanced_accuracy"],
            )
    metrics_frame = pd.DataFrame(fold_metrics)
    predictions_frame = pd.DataFrame(all_predictions)
    history_frame = pd.DataFrame(history)
    runtime_frame = pd.DataFrame(runtime_records)
    summary = summarize_metrics(metrics_frame)
    attention = pd.concat(attention_frames, ignore_index=True)
    attention["attention_rank"] = attention.groupby("sample_id")["attention_weight"].rank(
        method="first", ascending=False
    ).astype(int)
    attention = attention.merge(
        metadata[[
            "spectrum_id", "raw_file_stem", "scan_id", "retention_time_minutes",
            "precursor_mz", "precursor_charge", "has_psm", "psm_sequence",
            "psm_leading_protein", "psm_q_value",
        ]],
        on="spectrum_id", how="left", validate="many_to_one",
    )
    validate_stage_twelve_outputs(predictions_frame, attention, bags, variants)
    top_attention = attention[
        attention["attention_rank"].le(int(training["attention_top_k"]))
    ].sort_values(["sample_id", "attention_rank"])
    comparison = processed_comparison(
        config["processed_predictions"], predictions_frame,
        set(bags["sample_id"]), variants,
    )
    metrics_frame.to_csv(table_directory / "stage_12_fold_metrics.csv", index=False)
    summary.to_csv(table_directory / "stage_12_model_summary.csv", index=False)
    predictions_frame.to_csv(table_directory / "stage_12_predictions.csv", index=False)
    history_frame.to_csv(table_directory / "stage_12_training_history.csv", index=False)
    runtime_frame.to_csv(table_directory / "stage_12_runtime_memory.csv", index=False)
    attention.to_parquet(table_directory / "stage_12_attention_weights.parquet", index=False)
    top_attention.to_csv(table_directory / "stage_12_top_attention_spectra.csv", index=False)
    comparison.to_csv(table_directory / "stage_12_processed_comparison.csv", index=False)
    comparison_plot(comparison, figure_directory / "stage_12_model_comparison.png")
    learning_curve_plot(history_frame, figure_directory / "stage_12_learning_curves.png")
    bag_size_plot(metrics_frame, figure_directory / "stage_12_bag_size_sensitivity.png")
    elapsed = time.perf_counter() - started
    primary_metrics = metrics_frame[metrics_frame["analysis_type"].eq("primary_variant")]
    primary_summary = summarize_metrics(primary_metrics)
    best = primary_summary.iloc[0]
    sensitivity = metrics_frame[
        metrics_frame["analysis_type"].eq("bag_size_sensitivity")
    ].groupby("maximum_training_instances")["balanced_accuracy"].agg(["mean", "std"])
    result_lines = [
        f"- `{VARIANT_LABELS[row.experiment]}`: balanced accuracy {row.balanced_accuracy_mean:.3f} ± {row.balanced_accuracy_std:.3f}; AUROC {row.auroc_mean:.3f}; macro F1 {row.macro_f1_mean:.3f}."
        for row in primary_summary.itertuples(index=False)
    ]
    size_lines = [
        f"- {int(size)} spectra: balanced accuracy {row['mean']:.3f} ± {row['std']:.3f}."
        for size, row in sensitivity.iterrows()
    ]
    report_lines = [
        "# Stage 12 — Spectrum-aware MIL", "",
        "## Evaluation design", "",
        f"- Cells: {len(bags)} across {bags['raw_file_stem'].nunique()} raw-file groups.",
        "- Used the existing five grouped outer folds after removing only FP94AA/FP94AB cells; no raw file crosses a partition.",
        f"- Inner fold `{validation_inner_fold}` selected epochs using balanced accuracy; outer test cells were untouched until final evaluation.",
        f"- Training bags used {training['maximum_training_instances_per_bag']} stochastic spectra per epoch. Primary evaluation averaged {training['evaluation_repeats']} deterministic {training['maximum_evaluation_instances_per_bag']}-spectrum subsamples.",
        "- Numeric standardization and class weights were fitted within each training partition.",
        "- Cached Stage 11 HDF5 vectors were used; no mzML file was decoded during training.", "",
        "## Variants", "",
        "- Reporter + precursor metadata only.",
        "- Binned spectrum + reporter + precursor metadata.",
        "- Binned spectrum + reporter + precursor + PSM peptide/confidence; unidentified spectra use ID 0.",
        "- The optional pretrained spectral model remains deferred, as specified by the plan.", "",
        "## Primary grouped-fold results", "",
        *result_lines,
        f"- Best raw-spectrum variant: `{VARIANT_LABELS[str(best['experiment'])]}` at {best['balanced_accuracy_mean']:.3f} ± {best['balanced_accuracy_std']:.3f} balanced accuracy.", "",
        "## Bag-size sensitivity", "",
        *size_lines, "",
        "## Processed-data comparison", "",
        "- The comparison table evaluates every model on the same 202 outer-test cells.",
        "- Existing processed models were originally trained on 215 cells, whereas raw-spectrum models were trained on 202; this makes the comparison descriptive rather than an exact training-corpus-matched ablation.", "",
        "## Attention and resources", "",
        f"- Saved {len(attention):,} spectrum attention rows from the primary variant's first deterministic evaluation subsample and the top {training['attention_top_k']} spectra per cell.",
        "- Attention weights sum to one within each sampled cell bag and indicate model prioritization, not causal biological importance.",
        f"- Runtime: {elapsed / 60:.1f} minutes on `{device}`; peak process memory {peak_memory_mib():.1f} MiB.", "",
        "## Stage gate", "",
        "Stage 12 passes when cached training completes, every primary variant has exactly one OOF prediction per cell, grouped split checks pass, spectrum attention is saved, and the project test suite passes.", "",
    ]
    (report_directory / "stage_12_spectrum_mil.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    run_metadata = {
        "seed": seed,
        "device": str(device),
        "cells": int(len(bags)),
        "raw_file_groups": int(bags["raw_file_stem"].nunique()),
        "cached_spectra": int(len(metadata)),
        "source_spectrum_dimensions": source_spectrum_dim,
        "model_spectrum_dimensions": source_spectrum_dim // int(model_config["spectrum_downsample_factor"]),
        "psm_vocabulary_size_including_unknown": vocabulary_size,
        "primary_variants": variants,
        "bag_size_sensitivity": sensitivity_sizes,
        "runtime_seconds": elapsed,
        "process_peak_memory_mib": peak_memory_mib(),
        "best_primary_variant": str(best["experiment"]),
        "best_primary_balanced_accuracy_mean": float(best["balanced_accuracy_mean"]),
    }
    (table_directory / "stage_12_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    return run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spectrum_mil.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_12_spectrum_mil.log"))
    result = run_stage_twelve(args.config, args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 12 completed: %s", result)


if __name__ == "__main__":
    main()
