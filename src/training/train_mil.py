"""Train and evaluate Stage 8 peptide-level MIL models on grouped folds."""

from __future__ import annotations

import argparse
import copy
import json
import logging
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
from src.models.attention_mil import AttentionMIL
from src.models.deepsets import DeepSetsMIL
from src.training.collate import collate_peptide_bags
from src.training.datasets import (
    IntensityStandardizer,
    PeptideBagDataset,
    fit_intensity_standardizer,
    validate_mil_tables,
)
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging
from src.utils.seeds import seed_everything


MODEL_LABELS = {
    "mean_pooling": "Mean-pooling DeepSets",
    "max_pooling": "Max-pooling DeepSets",
    "attention": "Attention MIL",
}
METRIC_COLUMNS = [
    "balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc",
    "sensitivity", "specificity", "accuracy",
]


def resolve_device(name: str) -> torch.device:
    """Resolve a configured device without silently changing hardware."""
    normalized = name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def build_model(
    model_name: str,
    vocabulary_size: int,
    model_config: dict[str, Any],
) -> nn.Module:
    """Construct one configured MIL architecture using the real vocabulary size."""
    arguments = {
        "peptide_vocab_size": vocabulary_size,
        "peptide_embedding_dim": int(model_config["peptide_embedding_dim"]),
        "numeric_feature_dim": int(model_config["numeric_feature_dim"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "dropout": float(model_config["dropout"]),
    }
    if model_name == "attention":
        return AttentionMIL(**arguments)
    if model_name == "mean_pooling":
        return DeepSetsMIL(**arguments, pooling="mean")
    if model_name == "max_pooling":
        return DeepSetsMIL(**arguments, pooling="max")
    raise ValueError(f"unknown MIL model: {model_name}")


def make_loader(
    dataset: PeptideBagDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_peptide_bags,
        generator=generator,
    )


def positive_class_weight(bags: pd.DataFrame, sample_ids: list[str], device: torch.device) -> torch.Tensor:
    """Return negative/positive class weight fitted on training bags only."""
    labels = bags.set_index("sample_id").loc[sample_ids, "label_binary"].to_numpy(dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training data must contain both labels")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Train for one epoch and return bag-weighted mean loss."""
    model.train()
    loss_sum, bag_count = 0.0, 0
    for batch in loader:
        peptide_ids = batch["peptide_ids"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        bag_index = batch["bag_index"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(peptide_ids, numeric_features, bag_index)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        bag_count += len(labels)
    return loss_sum / bag_count


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_attention: bool = False,
) -> dict[str, Any]:
    """Evaluate bag predictions and optionally retain normalized attention weights."""
    model.eval()
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    sample_ids_all: list[str] = []
    attention_frames: list[pd.DataFrame] = []
    loss_sum, bag_count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            peptide_ids = batch["peptide_ids"].to(device)
            numeric_features = batch["numeric_features"].to(device)
            bag_index = batch["bag_index"].to(device)
            labels = batch["labels"].to(device)
            if collect_attention:
                if not isinstance(model, AttentionMIL):
                    raise TypeError("attention collection requires AttentionMIL")
                logits, weights = model(
                    peptide_ids, numeric_features, bag_index, return_attention=True
                )
            else:
                logits = model(peptide_ids, numeric_features, bag_index)
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
                    "instance_id": batch["instance_ids"].numpy(),
                    "peptide_id": batch["peptide_ids"].numpy(),
                    "attention_weight": weights.cpu().numpy(),
                }))
    y_true = np.concatenate(labels_all).astype(int)
    probabilities = np.concatenate(probabilities_all)
    predictions = (probabilities >= 0.5).astype(int)
    output = {
        "loss": loss_sum / bag_count,
        "sample_ids": sample_ids_all,
        "y_true": y_true,
        "y_pred": predictions,
        "probabilities": probabilities,
        "metrics": binary_metrics(y_true, predictions, probabilities),
    }
    output["attention"] = (
        pd.concat(attention_frames, ignore_index=True)
        if attention_frames else pd.DataFrame()
    )
    return output


def checkpoint_payload(
    model_name: str,
    model: nn.Module,
    outer_fold: int,
    seed: int,
    epoch: int,
    standardizer: IntensityStandardizer,
    training_ids: list[str],
    model_config: dict[str, Any],
    vocabulary_size: int,
    phase: str,
) -> dict[str, Any]:
    return {
        "stage": 8,
        "phase": phase,
        "model_name": model_name,
        "outer_fold": outer_fold,
        "seed": seed,
        "epochs": epoch,
        "vocabulary_size": vocabulary_size,
        "model_config": dict(model_config) | {"peptide_vocab_size": vocabulary_size},
        "intensity_standardizer": standardizer.as_dict(),
        "training_sample_ids": training_ids,
        "model_state_dict": model.state_dict(),
    }


def select_epoch_count(
    model_name: str,
    model: nn.Module,
    train_dataset: PeptideBagDataset,
    validation_dataset: PeptideBagDataset,
    training_ids: list[str],
    validation_ids: list[str],
    bags: pd.DataFrame,
    standardizer: IntensityStandardizer,
    outer_fold: int,
    seed: int,
    vocabulary_size: int,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    """Use untouched inner validation bags to select a final training duration."""
    batch_size = int(training_config["batch_size"])
    train_loader = make_loader(train_dataset, batch_size, True, seed)
    validation_loader = make_loader(validation_dataset, batch_size, False, seed)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weight(bags, training_ids, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    maximum_epochs = int(training_config["maximum_epochs"])
    minimum_epochs = int(training_config["minimum_epochs"])
    patience = int(training_config["early_stopping_patience"])
    minimum_delta = float(training_config["early_stopping_min_delta"])
    gradient_clip = float(training_config["gradient_clip_norm"])
    best_score, best_loss, best_epoch = -np.inf, np.inf, 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, maximum_epochs + 1):
        train_dataset.set_epoch(epoch)
        training_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, gradient_clip
        )
        validation = evaluate_model(model, validation_loader, criterion, device)
        score = float(validation["metrics"]["balanced_accuracy"])
        validation_loss = float(validation["loss"])
        history.append({
            "model": model_name,
            "outer_fold": outer_fold,
            "phase": "selection",
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in validation["metrics"].items()},
        })
        improved = score > best_score + minimum_delta or (
            abs(score - best_score) <= minimum_delta and validation_loss < best_loss
        )
        if improved:
            best_score, best_loss, best_epoch = score, validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= minimum_epochs and epochs_without_improvement >= patience:
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("early stopping failed to select a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        checkpoint_payload(
            model_name, model, outer_fold, seed, best_epoch, standardizer,
            training_ids, model_config, vocabulary_size, "selection",
        ) | {"validation_sample_ids": validation_ids, "best_validation_balanced_accuracy": best_score},
        checkpoint_path,
    )
    return best_epoch, best_score, history


def fit_final_model(
    model_name: str,
    model: nn.Module,
    dataset: PeptideBagDataset,
    training_ids: list[str],
    bags: pd.DataFrame,
    epochs: int,
    outer_fold: int,
    seed: int,
    training_config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Retrain on the complete outer-training partition for the selected duration."""
    loader = make_loader(dataset, int(training_config["batch_size"]), True, seed)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weight(bags, training_ids, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        training_loss = train_epoch(
            model, loader, optimizer, criterion, device,
            float(training_config["gradient_clip_norm"]),
        )
        history.append({
            "model": model_name,
            "outer_fold": outer_fold,
            "phase": "final_fit",
            "epoch": epoch,
            "training_loss": training_loss,
        })
    return history


def validate_split_coverage(
    bags: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    validation_inner_fold: int,
) -> None:
    """Reject split artifacts that do not cover all bags exactly as expected."""
    sample_ids = set(bags["sample_id"])
    if set(outer["sample_id"]) != sample_ids:
        raise ValueError("outer split samples differ from the Stage 7 bag index")
    test_counts = outer.loc[outer["partition"].eq("test"), "sample_id"].value_counts()
    if set(test_counts.index) != sample_ids or not test_counts.eq(1).all():
        raise ValueError("each sample must occur in exactly one outer test fold")
    for outer_fold, outer_frame in outer.groupby("outer_fold"):
        if outer_frame.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
            raise ValueError(f"a raw file crosses outer fold {outer_fold} partitions")
        training_ids = set(outer_frame.loc[outer_frame["partition"].eq("train"), "sample_id"])
        selected = inner[
            inner["outer_fold"].eq(outer_fold)
            & inner["inner_fold"].eq(validation_inner_fold)
        ]
        if set(selected["sample_id"]) != training_ids:
            raise ValueError(f"inner split does not cover outer fold {outer_fold} training bags")
        if selected.groupby("raw_file_stem")["partition"].nunique().gt(1).any():
            raise ValueError(f"a raw file crosses the selected inner split for outer fold {outer_fold}")
        if selected.groupby("partition")["label_raw"].nunique().lt(2).any():
            raise ValueError(f"selected inner split lacks both labels for outer fold {outer_fold}")


def comparison_plot(
    mil_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare MIL models with the Stage 6 peptide-logistic reference."""
    baseline = baseline_metrics[baseline_metrics["model"].eq("peptide_logistic")].copy()
    baseline["model"] = "peptide_logistic"
    combined = pd.concat([baseline, mil_metrics], ignore_index=True)
    order = ["peptide_logistic", *MODEL_LABELS]
    labels = {"peptide_logistic": "Peptide logistic", **MODEL_LABELS}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for axis, metric in zip(axes, ["balanced_accuracy", "macro_f1"]):
        for position, model_name in enumerate(order):
            values = combined.loc[combined["model"].eq(model_name), metric].to_numpy()
            axis.scatter(np.full(len(values), position), values, s=36, alpha=0.75)
            axis.plot([position - 0.18, position + 0.18], [values.mean(), values.mean()], linewidth=2)
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(order)), [labels[name] for name in order], rotation=24, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(f"{metric.replace('_', ' ').title()} across grouped folds")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def learning_curve_plot(history: pd.DataFrame, output_path: Path) -> None:
    """Plot inner-validation balanced accuracy used for early stopping."""
    selection = history[history["phase"].eq("selection")]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True, constrained_layout=True)
    for axis, model_name in zip(axes, MODEL_LABELS):
        model_history = selection[selection["model"].eq(model_name)]
        for outer_fold, fold_history in model_history.groupby("outer_fold"):
            axis.plot(
                fold_history["epoch"],
                fold_history["validation_balanced_accuracy"],
                marker="o", markersize=2.5, linewidth=1,
                label=f"Fold {outer_fold}",
            )
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Epoch")
        axis.set_title(MODEL_LABELS[model_name])
    axes[0].set_ylabel("Inner-validation balanced accuracy")
    axes[-1].legend(fontsize=8, loc="lower right")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def dry_run_forward(
    config: dict[str, Any],
    instances: pd.DataFrame,
    bags: pd.DataFrame,
    vocabulary: pd.DataFrame,
    outer: pd.DataFrame,
    device: torch.device,
) -> dict[str, Any]:
    """Validate a real-data forward pass for every requested architecture."""
    fold = outer[outer["outer_fold"].eq(outer["outer_fold"].min())]
    training_ids = fold.loc[fold["partition"].eq("train"), "sample_id"].tolist()
    standardizer = fit_intensity_standardizer(instances, bags, training_ids)
    dataset = PeptideBagDataset(
        instances, bags, training_ids[:2], standardizer,
        maximum_instances_per_bag=32, seed=int(config["seed"]),
    )
    batch = collate_peptide_bags([dataset[0], dataset[1]])
    vocabulary_size = int(vocabulary["peptide_id"].max()) + 1
    output_shapes = {}
    for model_name in config["training"]["models"]:
        model = build_model(model_name, vocabulary_size, config["model"]).to(device).eval()
        with torch.no_grad():
            logits = model(
                batch["peptide_ids"].to(device),
                batch["numeric_features"].to(device),
                batch["bag_index"].to(device),
            )
        output_shapes[model_name] = list(logits.shape)
    return {
        "dry_run": True,
        "bags": len(bags),
        "instances": len(instances),
        "vocabulary_size_including_reserved_id": vocabulary_size,
        "model_output_shapes": output_shapes,
    }


def run_stage_eight(
    config_path: str | Path = "configs/mil.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train all Stage 8 MIL models and save grouped-fold evaluation artifacts."""
    config = load_yaml_config(config_path)
    training_config = config["training"]
    if int(config["model"]["numeric_feature_dim"]) != 1:
        raise ValueError("Stage 8 currently supplies one numeric feature: standardized log2 intensity")
    bag_directory = Path(config.get("data", {}).get("output_directory", "data/processed/mil_bags"))
    vocabulary = pd.read_parquet(bag_directory / "peptide_vocabulary.parquet")
    instances = pd.read_parquet(bag_directory / "peptide_instances.parquet")
    bags = pd.read_parquet(bag_directory / "bag_index.parquet")
    outer = pd.read_csv("data/splits/outer_fold_assignments.csv")
    inner = pd.read_csv("data/splits/inner_fold_assignments.csv")
    validate_mil_tables(instances, bags, vocabulary)
    validation_inner_fold = int(training_config["validation_inner_fold"])
    validate_split_coverage(bags, outer, inner, validation_inner_fold)
    requested_models = [str(name) for name in training_config["models"]]
    unknown_models = sorted(set(requested_models) - set(MODEL_LABELS))
    if unknown_models:
        raise ValueError(f"unknown configured MIL models: {unknown_models}")
    device = resolve_device(str(training_config.get("device", "cpu")))
    seed = int(config["seed"])
    seed_everything(seed)
    vocabulary_size = int(vocabulary["peptide_id"].max()) + 1
    if dry_run:
        return dry_run_forward(config, instances, bags, vocabulary, outer, device)
    checkpoint_directory = Path("outputs/checkpoints/stage_08")
    table_directory = Path("outputs/tables")
    figure_directory = Path("outputs/figures")
    report_directory = Path("outputs/reports")
    for directory in (checkpoint_directory, table_directory, figure_directory, report_directory):
        directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("single_cell_proteomics_mil")
    label_by_sample = bags.set_index("sample_id")["label_raw"]
    maximum_instances = int(training_config["maximum_training_instances_per_bag"])
    all_predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    attention_frames: list[pd.DataFrame] = []
    started = time.perf_counter()
    for outer_fold in sorted(outer["outer_fold"].unique()):
        outer_frame = outer[outer["outer_fold"].eq(outer_fold)]
        outer_training_ids = outer_frame.loc[outer_frame["partition"].eq("train"), "sample_id"].tolist()
        test_ids = outer_frame.loc[outer_frame["partition"].eq("test"), "sample_id"].tolist()
        selected_inner = inner[
            inner["outer_fold"].eq(outer_fold)
            & inner["inner_fold"].eq(validation_inner_fold)
        ]
        selection_training_ids = selected_inner.loc[selected_inner["partition"].eq("train"), "sample_id"].tolist()
        validation_ids = selected_inner.loc[selected_inner["partition"].eq("validation"), "sample_id"].tolist()
        for model_index, model_name in enumerate(requested_models):
            model_seed = seed + int(outer_fold) * 1_000 + model_index * 10_000
            logger.info("Stage 8 | fold=%s model=%s | selecting epoch count", outer_fold, model_name)
            seed_everything(model_seed)
            selection_standardizer = fit_intensity_standardizer(
                instances, bags, selection_training_ids
            )
            selection_training_dataset = PeptideBagDataset(
                instances, bags, selection_training_ids, selection_standardizer,
                maximum_instances_per_bag=maximum_instances, seed=model_seed,
            )
            validation_dataset = PeptideBagDataset(
                instances, bags, validation_ids, selection_standardizer,
                maximum_instances_per_bag=None, seed=model_seed,
            )
            selection_model = build_model(model_name, vocabulary_size, config["model"]).to(device)
            best_epoch, best_validation_score, selection_history = select_epoch_count(
                model_name, selection_model, selection_training_dataset, validation_dataset,
                selection_training_ids, validation_ids, bags, selection_standardizer,
                int(outer_fold), model_seed, vocabulary_size, config["model"],
                training_config,
                checkpoint_directory / f"{model_name}_fold_{outer_fold}_selection.pt",
                device,
            )
            all_history.extend(selection_history)
            logger.info(
                "Stage 8 | fold=%s model=%s | best_epoch=%s validation_balanced_accuracy=%.3f",
                outer_fold, model_name, best_epoch, best_validation_score,
            )
            seed_everything(model_seed)
            final_standardizer = fit_intensity_standardizer(instances, bags, outer_training_ids)
            final_training_dataset = PeptideBagDataset(
                instances, bags, outer_training_ids, final_standardizer,
                maximum_instances_per_bag=maximum_instances, seed=model_seed,
            )
            test_dataset = PeptideBagDataset(
                instances, bags, test_ids, final_standardizer,
                maximum_instances_per_bag=None, seed=model_seed,
            )
            final_model = build_model(model_name, vocabulary_size, config["model"]).to(device)
            all_history.extend(fit_final_model(
                model_name, final_model, final_training_dataset, outer_training_ids,
                bags, best_epoch, int(outer_fold), model_seed, training_config, device,
            ))
            test_loader = make_loader(
                test_dataset, int(training_config["batch_size"]), False, model_seed
            )
            test_criterion = nn.BCEWithLogitsLoss(
                pos_weight=positive_class_weight(bags, outer_training_ids, device)
            )
            test = evaluate_model(
                final_model, test_loader, test_criterion, device,
                collect_attention=model_name == "attention",
            )
            torch.save(
                checkpoint_payload(
                    model_name, final_model, int(outer_fold), model_seed, best_epoch,
                    final_standardizer, outer_training_ids, config["model"],
                    vocabulary_size, "final_fit",
                ) | {"test_sample_ids": test_ids},
                checkpoint_directory / f"{model_name}_fold_{outer_fold}_final.pt",
            )
            fold_metrics.append({
                "model": model_name,
                "outer_fold": int(outer_fold),
                "test_samples": len(test_ids),
                "selected_epochs": best_epoch,
                "best_inner_balanced_accuracy": best_validation_score,
                **test["metrics"],
            })
            for sample_id, y_true, y_pred, probability in zip(
                test["sample_ids"], test["y_true"], test["y_pred"], test["probabilities"]
            ):
                all_predictions.append({
                    "model": model_name,
                    "outer_fold": int(outer_fold),
                    "sample_id": sample_id,
                    "label_raw": label_by_sample.loc[sample_id],
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "probability_sc_u": float(probability),
                })
            if model_name == "attention":
                attention = test["attention"].copy()
                attention["outer_fold"] = int(outer_fold)
                attention_frames.append(attention)
            logger.info(
                "Stage 8 | fold=%s model=%s | test_balanced_accuracy=%.3f",
                outer_fold, model_name, test["metrics"]["balanced_accuracy"],
            )
    metrics_frame = pd.DataFrame(fold_metrics)
    predictions_frame = pd.DataFrame(all_predictions)
    history_frame = pd.DataFrame(all_history)
    summary = metrics_frame.groupby("model")[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index().sort_values("balanced_accuracy_mean", ascending=False)
    attention_frame = pd.concat(attention_frames, ignore_index=True)
    attention_frame["attention_rank"] = (
        attention_frame.groupby("sample_id")["attention_weight"]
        .rank(method="first", ascending=False).astype(int)
    )
    log_intensity_by_instance = instances.set_index("instance_id")["log2_intensity"]
    attention_frame["log2_intensity"] = attention_frame["instance_id"].map(log_intensity_by_instance)
    top_attention = attention_frame[attention_frame["attention_rank"].le(20)].merge(
        vocabulary, on="peptide_id", how="left", validate="many_to_one"
    ).sort_values(["sample_id", "attention_rank"])
    metrics_frame.to_csv(table_directory / "stage_08_fold_metrics.csv", index=False)
    summary.to_csv(table_directory / "stage_08_model_summary.csv", index=False)
    predictions_frame.to_csv(table_directory / "stage_08_predictions.csv", index=False)
    history_frame.to_csv(table_directory / "stage_08_training_history.csv", index=False)
    attention_frame.to_parquet(table_directory / "stage_08_attention_weights.parquet", index=False)
    top_attention.to_csv(table_directory / "stage_08_top_attention_peptides.csv", index=False)
    baseline_metrics = pd.read_csv(table_directory / "stage_06_fold_metrics.csv")
    comparison_plot(
        metrics_frame, baseline_metrics,
        figure_directory / "stage_08_mil_comparison.png",
    )
    learning_curve_plot(history_frame, figure_directory / "stage_08_learning_curves.png")
    elapsed_seconds = time.perf_counter() - started
    best = summary.iloc[0]
    baseline_mean = float(
        baseline_metrics.loc[
            baseline_metrics["model"].eq("peptide_logistic"), "balanced_accuracy"
        ].mean()
    )
    model_lines = [
        f"- `{MODEL_LABELS[row.model]}`: balanced accuracy {row.balanced_accuracy_mean:.3f} ± {row.balanced_accuracy_std:.3f}; AUROC {row.auroc_mean:.3f}; Macro F1 {row.macro_f1_mean:.3f}."
        for row in summary.itertuples(index=False)
    ]
    (report_directory / "stage_08_peptide_mil_models.md").write_text("\n".join([
        "# Stage 8 — Peptide-level MIL models", "",
        "## Evaluation design", "",
        "- Models use the immutable five raw-file-grouped outer folds from Stage 5.",
        f"- Inner fold `{validation_inner_fold}` selects the epoch count by validation balanced accuracy.",
        "- After selection, each model is reinitialized and fitted on the complete outer-training partition for the selected number of epochs.",
        "- Outer test bags are not used for normalization, early stopping, or model selection.",
        f"- Training bags are stochastically capped at {maximum_instances} instances per epoch; validation and test bags use every observed instance.",
        "- Log2 intensity standardization and class weights are fitted separately inside each training partition.", "",
        f"- This proof-of-concept run uses one deterministic training seed (`{seed}`).", "",
        "## Models", "",
        "- Mean-pooling DeepSets.",
        "- Max-pooling DeepSets.",
        "- Gated-attention MIL.", "",
        "## Results", "",
        *model_lines,
        f"- Stage 6 peptide-logistic reference balanced accuracy: {baseline_mean:.3f}.",
        f"- Best Stage 8 model: `{MODEL_LABELS[str(best['model'])]}` at {best['balanced_accuracy_mean']:.3f} ± {best['balanced_accuracy_std']:.3f}.",
        "- This comparison is descriptive; paired confidence intervals and formal model comparison belong to Stage 9.", "",
        "## Integrity", "",
        f"- Test predictions: {len(predictions_frame)} ({len(bags)} per model).",
        f"- Attention weights: {len(attention_frame)} (one for every observed peptide instance when its cell was in an outer test fold).",
        "- Attention weights sum to one within every cell and are model-prioritization scores, not causal biomarker evidence.",
        f"- Runtime: {elapsed_seconds / 60:.1f} minutes on `{device}`.", "",
    ]), encoding="utf-8")
    run_metadata = {
        "seed": seed,
        "device": str(device),
        "models": requested_models,
        "outer_folds": int(outer["outer_fold"].nunique()),
        "validation_inner_fold": validation_inner_fold,
        "vocabulary_size_including_reserved_id": vocabulary_size,
        "bags": int(len(bags)),
        "instances": int(len(instances)),
        "runtime_seconds": elapsed_seconds,
        "best_model": str(best["model"]),
        "best_balanced_accuracy": float(best["balanced_accuracy_mean"]),
    }
    (table_directory / "stage_08_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    return run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mil.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_08_mil.log"))
    result = run_stage_eight(args.config, args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 8 completed: %s", result)


if __name__ == "__main__":
    main()
