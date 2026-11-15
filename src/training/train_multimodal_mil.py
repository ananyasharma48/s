"""Execute Stage 16 protein-plus-spectrum multimodal MIL."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.bootstrap import (
    DEFAULT_METRICS,
    group_bootstrap_metrics,
    paired_group_bootstrap_differences,
    percentile_interval,
)
from src.evaluation.calibration import calibration_metrics, reliability_table
from src.evaluation.metrics import binary_metrics
from src.models.classical import fixed_vector_logistic
from src.models.multimodal_mil import (
    JointMultimodalMIL,
    ResidualMultimodalMIL,
    SpectrumBagEncoder,
)
from src.training.spectrum_datasets import (
    SpectrumArrayStore,
    SpectrumBagDataset,
    VectorStandardizer,
    collate_spectrum_bags,
    fit_numeric_standardizer,
    make_psm_vocabulary,
    validate_spectrum_bags,
)
from src.training.train_mil import resolve_device
from src.training.train_spectrum_mil import (
    build_spectrum_model,
    evaluate_once as evaluate_spectrum_once,
    make_loader as make_spectrum_loader,
)
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging
from src.utils.seeds import seed_everything


MODEL_LABELS = {
    "protein_logistic": "Protein logistic",
    "spectrum_psm": "Spectrum + reporter + PSM",
    "late_fusion": "Validation-weighted late fusion",
    "joint_hybrid": "Joint protein-spectrum MIL",
    "residual_hybrid": "Protein-initialized residual MIL",
}
HYBRID_MODELS = ("late_fusion", "joint_hybrid", "residual_hybrid")
METRIC_COLUMNS = [
    "balanced_accuracy", "macro_f1", "auroc", "auprc", "mcc",
    "sensitivity", "specificity", "accuracy",
]
PROGRESS_NAMES = ("predictions", "fold_metrics", "history", "runtime", "tuning")


@dataclass(frozen=True)
class ProteinPreprocessor:
    """Training-partition protein filtering, imputation, and standardization."""

    support: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    standard_deviation: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values, dtype=float)[:, self.support]
        selected = np.log1p(selected)
        selected = np.where(np.isnan(selected), self.median, selected)
        return ((selected - self.mean) / self.standard_deviation).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        return {
            "support_indices": np.flatnonzero(self.support).astype(int).tolist(),
            "median": self.median.astype(float).tolist(),
            "mean": self.mean.astype(float).tolist(),
            "standard_deviation": self.standard_deviation.astype(float).tolist(),
        }


def fit_protein_preprocessor(
    protein_matrix: pd.DataFrame, sample_ids: list[str], minimum_prevalence: float,
) -> ProteinPreprocessor:
    values = protein_matrix.reindex(columns=sample_ids).T.to_numpy(dtype=float)
    support = np.mean(~np.isnan(values), axis=0) >= minimum_prevalence
    if not np.any(support):
        raise ValueError("protein prevalence filtering removed every feature")
    transformed = np.log1p(values[:, support])
    median = np.nanmedian(transformed, axis=0)
    transformed = np.where(np.isnan(transformed), median, transformed)
    mean = transformed.mean(axis=0)
    standard_deviation = transformed.std(axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return ProteinPreprocessor(support, median, mean, standard_deviation)


class MultimodalDataset(Dataset):
    """Align one transformed protein vector with one spectrum bag per cell."""

    def __init__(
        self, spectrum_dataset: SpectrumBagDataset, protein_matrix: pd.DataFrame,
        sample_ids: list[str], protein_preprocessor: ProteinPreprocessor,
    ) -> None:
        self.spectrum_dataset = spectrum_dataset
        self.sample_ids = list(sample_ids)
        self.protein = protein_preprocessor.transform(
            protein_matrix.reindex(columns=sample_ids).T.to_numpy(dtype=float)
        )
        if len(self.spectrum_dataset) != len(self.sample_ids):
            raise ValueError("protein vectors and spectrum bags have different lengths")

    def set_epoch(self, epoch: int) -> None:
        self.spectrum_dataset.set_epoch(epoch)

    def close(self) -> None:
        self.spectrum_dataset.close()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.spectrum_dataset[index]
        if item["sample_id"] != self.sample_ids[index]:
            raise ValueError("protein and spectrum sample ordering differs")
        item["protein_features"] = torch.from_numpy(self.protein[index].copy())
        return item


def collate_multimodal(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_spectrum_bags(items)
    batch["protein_features"] = torch.stack([item["protein_features"] for item in items])
    return batch


def make_loader(dataset: MultimodalDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_multimodal, generator=torch.Generator().manual_seed(seed),
    )


def forward_model(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return model(
        batch["protein_features"].to(device),
        batch["spectrum_features"].to(device),
        batch["numeric_features"].to(device),
        batch["peptide_ids"].to(device),
        batch["bag_index"].to(device),
    )


def positive_class_weight(bags: pd.DataFrame, sample_ids: list[str], device: torch.device) -> torch.Tensor:
    labels = bags.set_index("sample_id").loc[sample_ids, "label_binary"].to_numpy(int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("training partition must contain both labels")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def build_spectrum_encoder(
    spectrum_dim: int, vocabulary_size: int, config: dict[str, Any]
) -> SpectrumBagEncoder:
    return SpectrumBagEncoder(
        spectrum_feature_dim=spectrum_dim,
        numeric_feature_dim=8,
        hidden_dim=int(config["hidden_dim"]),
        spectrum_projection_dim=int(config["spectrum_projection_dim"]),
        peptide_vocab_size=vocabulary_size,
        peptide_embedding_dim=int(config["peptide_embedding_dim"]),
        dropout=float(config["dropout"]),
    )


def build_hybrid_model(
    variant: str, protein_dim: int, spectrum_dim: int, vocabulary_size: int,
    config: dict[str, Any], protein_logistic: LogisticRegression | None = None,
) -> nn.Module:
    encoder = build_spectrum_encoder(spectrum_dim, vocabulary_size, config["spectrum"])
    if variant == "joint_hybrid":
        return JointMultimodalMIL(
            protein_dim, int(config["protein"]["hidden_dim"]), encoder,
            float(config["spectrum"]["dropout"]),
        )
    if variant == "residual_hybrid":
        if protein_logistic is None:
            raise ValueError("residual fusion requires a fitted protein logistic model")
        return ResidualMultimodalMIL(
            protein_dim, encoder,
            torch.tensor(protein_logistic.coef_[0], dtype=torch.float32),
            torch.tensor(protein_logistic.intercept_, dtype=torch.float32),
            float(config["spectrum"]["dropout"]),
        )
    raise ValueError(f"unknown hybrid variant: {variant}")


def fit_residual_baseline(
    transformed_protein: np.ndarray, labels: np.ndarray, c_value: float, seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=3000,
        solver="liblinear", random_state=seed,
    )
    return model.fit(transformed_protein, labels)


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: nn.Module, device: torch.device, gradient_clip_norm: float,
) -> float:
    model.train()
    loss_sum, sample_count = 0.0, 0
    for batch in loader:
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        sample_count += len(labels)
    return loss_sum / sample_count


def evaluate_once(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
) -> dict[str, Any]:
    model.eval()
    sample_ids: list[str] = []
    truths, probabilities = [], []
    loss_sum, sample_count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = forward_model(model, batch, device)
            loss = criterion(logits, labels)
            truths.append(labels.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            sample_ids.extend(batch["sample_ids"])
            loss_sum += float(loss) * len(labels)
            sample_count += len(labels)
    y_true = np.concatenate(truths).astype(int)
    probability = np.concatenate(probabilities)
    prediction = (probability >= 0.5).astype(int)
    return {
        "sample_ids": sample_ids, "y_true": y_true, "y_pred": prediction,
        "probabilities": probability, "loss": loss_sum / sample_count,
        "metrics": binary_metrics(y_true, prediction, probability),
    }


def evaluate_repeated(
    model: nn.Module, dataset: MultimodalDataset, criterion: nn.Module,
    device: torch.device, batch_size: int, seed: int, repeats: int,
) -> dict[str, Any]:
    results = []
    for repeat in range(repeats):
        dataset.set_epoch(10_000 + repeat)
        results.append(evaluate_once(
            model, make_loader(dataset, batch_size, False, seed + repeat), criterion, device
        ))
    first = results[0]
    for result in results[1:]:
        if result["sample_ids"] != first["sample_ids"] or not np.array_equal(result["y_true"], first["y_true"]):
            raise ValueError("repeated multimodal evaluations are misaligned")
    probability = np.mean([result["probabilities"] for result in results], axis=0)
    prediction = (probability >= 0.5).astype(int)
    return {
        "sample_ids": first["sample_ids"], "y_true": first["y_true"],
        "y_pred": prediction, "probabilities": probability,
        "metrics": binary_metrics(first["y_true"], prediction, probability),
    }


def select_epochs(
    model: nn.Module, training_dataset: MultimodalDataset,
    validation_dataset: MultimodalDataset, bags: pd.DataFrame,
    training_ids: list[str], config: dict[str, Any], device: torch.device,
    model_name: str, seed: int, outer_fold: int,
) -> tuple[int, float, list[dict[str, Any]], dict[str, torch.Tensor]]:
    training_loader = make_loader(training_dataset, int(config["batch_size"]), True, seed)
    validation_loader = make_loader(validation_dataset, int(config["evaluation_batch_size"]), False, seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(bags, training_ids, device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    best_score, best_loss, best_epoch = -np.inf, np.inf, 0
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history = []
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        training_dataset.set_epoch(epoch)
        train_loss = train_epoch(
            model, training_loader, optimizer, criterion, device,
            float(config["gradient_clip_norm"]),
        )
        validation_dataset.set_epoch(0)
        result = evaluate_once(model, validation_loader, criterion, device)
        score, loss = float(result["metrics"]["balanced_accuracy"]), float(result["loss"])
        history.append({
            "model": model_name, "seed": seed, "outer_fold": outer_fold,
            "phase": "selection", "epoch": epoch, "training_loss": train_loss,
            "validation_loss": loss,
            **{f"validation_{key}": value for key, value in result["metrics"].items()},
        })
        improved = score > best_score + float(config["early_stopping_min_delta"]) or (
            abs(score - best_score) <= float(config["early_stopping_min_delta"]) and loss < best_loss
        )
        if improved:
            best_score, best_loss, best_epoch = score, loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch >= int(config["minimum_epochs"]) and without_improvement >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("hybrid early stopping did not select a checkpoint")
    return best_epoch, best_score, history, best_state


def fit_final(
    model: nn.Module, dataset: MultimodalDataset, bags: pd.DataFrame,
    training_ids: list[str], epochs: int, config: dict[str, Any],
    device: torch.device, model_name: str, seed: int, outer_fold: int,
) -> list[dict[str, Any]]:
    loader = make_loader(dataset, int(config["batch_size"]), True, seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(bags, training_ids, device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        loss = train_epoch(
            model, loader, optimizer, criterion, device, float(config["gradient_clip_norm"])
        )
        history.append({
            "model": model_name, "seed": seed, "outer_fold": outer_fold,
            "phase": "final_fit", "epoch": epoch, "training_loss": loss,
        })
    return history


def permutation_invariance_error(
    model: nn.Module, batch: dict[str, Any], device: torch.device,
) -> float:
    """Compare logits before and after reversing instances within every bag."""
    model.eval()
    bag_index = batch["bag_index"]
    order_parts = []
    for bag_id in range(int(bag_index.max()) + 1):
        order_parts.append(torch.nonzero(bag_index == bag_id, as_tuple=False).flatten().flip(0))
    order = torch.cat(order_parts)
    shuffled = dict(batch)
    for key in ("spectrum_features", "numeric_features", "peptide_ids", "spectrum_ids", "reporter_intensity"):
        shuffled[key] = batch[key][order]
    shuffled["bag_index"] = batch["bag_index"][order]
    with torch.no_grad():
        original = forward_model(model, batch, device)
        permuted = forward_model(model, shuffled, device)
    return float(torch.max(torch.abs(original - permuted)).cpu())


def select_fusion_weight(
    y_true: np.ndarray, protein_probability: np.ndarray,
    spectrum_probability: np.ndarray, weights: list[float],
) -> tuple[float, float]:
    """Choose protein weight by validation balanced accuracy, preferring protein on ties."""
    best_weight, best_score = 1.0, -np.inf
    for weight in sorted((float(value) for value in weights), reverse=True):
        probability = weight * protein_probability + (1.0 - weight) * spectrum_probability
        prediction = (probability >= 0.5).astype(int)
        score = binary_metrics(y_true, prediction, probability)["balanced_accuracy"]
        if score > best_score + 1e-12:
            best_weight, best_score = weight, score
    return best_weight, best_score


def load_progress(directory: Path, resume: bool) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name in PROGRESS_NAMES}
    if resume:
        for name in PROGRESS_NAMES:
            path = directory / f"stage_16_{name}.partial.csv"
            if path.exists() and path.stat().st_size:
                result[name] = pd.read_csv(path).to_dict("records")
    return result


def save_progress(directory: Path, progress: dict[str, list[dict[str, Any]]]) -> None:
    for name in PROGRESS_NAMES:
        frame = pd.DataFrame(progress[name])
        if not frame.empty:
            frame.to_csv(directory / f"stage_16_{name}.partial.csv", index=False)


def complete(
    progress: dict[str, list[dict[str, Any]]], model: str, seed: int,
    outer_fold: int, expected: int,
) -> bool:
    frame = pd.DataFrame(progress["predictions"])
    if frame.empty:
        return False
    selected = frame[
        frame["model"].eq(model) & frame["seed"].eq(seed)
        & frame["outer_fold"].eq(outer_fold)
    ]
    return len(selected) == expected and selected["sample_id"].nunique() == expected


def add_prediction_rows(
    target: list[dict[str, Any]], model: str, seed: int, outer_fold: int,
    sample_ids: list[str], label_by_sample: pd.Series, y_true: np.ndarray,
    y_pred: np.ndarray, probability: np.ndarray,
) -> None:
    for sample_id, truth, prediction, score in zip(sample_ids, y_true, y_pred, probability):
        target.append({
            "model": model, "model_label": MODEL_LABELS[model], "seed": seed,
            "outer_fold": outer_fold, "sample_id": sample_id,
            "label_raw": label_by_sample.loc[sample_id], "y_true": int(truth),
            "y_pred": int(prediction), "probability_sc_u": float(score),
        })


def peak_memory_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def build_multimodal_dataset(
    bags: pd.DataFrame, metadata: pd.DataFrame, sample_ids: list[str],
    spectra_path: Path, reporter_path: Path, spectrum_standardizer: VectorStandardizer,
    peptide_to_id: dict[str, int], maximum_instances: int,
    seed: int, array_store: SpectrumArrayStore, protein_matrix: pd.DataFrame,
    protein_preprocessor: ProteinPreprocessor, downsample_factor: int,
) -> MultimodalDataset:
    spectrum = SpectrumBagDataset(
        bags, metadata, sample_ids, spectra_path, reporter_path,
        spectrum_standardizer, peptide_to_id, True, True, downsample_factor,
        maximum_instances, seed, array_store,
    )
    return MultimodalDataset(spectrum, protein_matrix, sample_ids, protein_preprocessor)


def run_late_fusion(
    config: dict[str, Any], manifest: pd.DataFrame, protein_matrix: pd.DataFrame,
    bags: pd.DataFrame, metadata: pd.DataFrame, outer: pd.DataFrame,
    spectra_path: Path, reporter_path: Path, array_store: SpectrumArrayStore,
    peptide_to_id: dict[str, int], vocabulary_size: int,
    stage15_predictions: pd.DataFrame, stage15_tuning: pd.DataFrame,
    checkpoint_directory: Path, table_directory: Path,
    progress: dict[str, list[dict[str, Any]]], device: torch.device,
) -> None:
    logger = logging.getLogger("single_cell_proteomics_mil")
    label_by_sample = manifest.set_index("sample_id")["label_raw"]
    y_by_sample = label_by_sample.eq(config["positive_label"]).astype(int)
    stage15_checkpoint_directory = Path(config["data"]["stage_15_checkpoints"])
    spectrum_variant = str(config["late_fusion"]["spectrum_model"])
    downsample = int(config["spectrum"]["spectrum_downsample_factor"])
    source_dim = array_store.spectrum_features.shape[1] * downsample
    for outer_fold in sorted(outer["outer_fold"].unique()):
        fold = outer[outer["outer_fold"].eq(outer_fold)]
        test_ids = fold.loc[fold["partition"].eq("test"), "sample_id"].tolist()
        c_value = float(stage15_tuning[
            stage15_tuning["model"].eq("protein_logistic")
            & stage15_tuning["outer_fold"].eq(outer_fold)
        ]["selected_c"].iloc[0])
        for seed in (int(value) for value in config["neural_seeds"]):
            if complete(progress, "late_fusion", seed, int(outer_fold), len(test_ids)):
                continue
            started = time.perf_counter()
            logger.info("Stage 16 | fold=%s seed=%s late_fusion", outer_fold, seed)
            checkpoint = torch.load(
                stage15_checkpoint_directory / f"{spectrum_variant}_seed_{seed}_fold_{outer_fold}_selection.pt",
                map_location="cpu", weights_only=False,
            )
            selection_ids = [str(value) for value in checkpoint["training_sample_ids"]]
            validation_ids = [str(value) for value in checkpoint["validation_sample_ids"]]
            protein_model = fixed_vector_logistic(
                float(config["protein"]["minimum_feature_prevalence"]), seed
            ).set_params(model__C=c_value)
            protein_model.fit(
                protein_matrix.reindex(columns=selection_ids).T,
                y_by_sample.loc[selection_ids].to_numpy(int),
            )
            validation_protein = protein_model.predict_proba(
                protein_matrix.reindex(columns=validation_ids).T
            )[:, 1]
            standardizer = VectorStandardizer(
                np.asarray(checkpoint["numeric_standardizer"]["mean"], dtype=np.float32),
                np.asarray(checkpoint["numeric_standardizer"]["standard_deviation"], dtype=np.float32),
            )
            run_seed = seed + int(outer_fold) * 1_000 + 20_000
            validation_dataset = SpectrumBagDataset(
                bags, metadata, validation_ids, spectra_path, reporter_path,
                standardizer, peptide_to_id, True, True, downsample,
                int(config["training"]["maximum_evaluation_instances_per_bag"]),
                run_seed, array_store,
            )
            spectrum_model = build_spectrum_model(
                spectrum_variant, source_dim, vocabulary_size, checkpoint["model_config"]
            ).to(device)
            spectrum_model.load_state_dict(checkpoint["model_state_dict"])
            validation_result = evaluate_spectrum_once(
                spectrum_model,
                make_spectrum_loader(
                    validation_dataset, int(config["training"]["evaluation_batch_size"]),
                    False, run_seed,
                ),
                nn.BCEWithLogitsLoss(), device,
            )
            validation_dataset.close()
            if validation_result["sample_ids"] != validation_ids:
                raise ValueError("late-fusion validation predictions are misaligned")
            weight, validation_score = select_fusion_weight(
                y_by_sample.loc[validation_ids].to_numpy(int), validation_protein,
                validation_result["probabilities"], config["late_fusion"]["protein_weight_grid"],
            )
            test_protein = stage15_predictions[
                stage15_predictions["model"].eq("protein_logistic")
                & stage15_predictions["outer_fold"].eq(outer_fold)
            ].set_index("sample_id").loc[test_ids, "probability_sc_u"].to_numpy(float)
            test_spectrum = stage15_predictions[
                stage15_predictions["model"].eq(spectrum_variant)
                & stage15_predictions["seed"].eq(seed)
                & stage15_predictions["outer_fold"].eq(outer_fold)
            ].set_index("sample_id").loc[test_ids, "probability_sc_u"].to_numpy(float)
            probability = weight * test_protein + (1.0 - weight) * test_spectrum
            prediction = (probability >= 0.5).astype(int)
            truth = y_by_sample.loc[test_ids].to_numpy(int)
            add_prediction_rows(
                progress["predictions"], "late_fusion", seed, int(outer_fold),
                test_ids, label_by_sample, truth, prediction, probability,
            )
            metrics = binary_metrics(truth, prediction, probability)
            progress["fold_metrics"].append({
                "model": "late_fusion", "model_label": MODEL_LABELS["late_fusion"],
                "seed": seed, "outer_fold": int(outer_fold), "test_samples": len(test_ids),
                "selected_epochs": np.nan, "best_inner_balanced_accuracy": validation_score,
                "permutation_invariance_max_abs_error": 0.0, **metrics,
            })
            progress["tuning"].append({
                "model": "late_fusion", "seed": seed, "outer_fold": int(outer_fold),
                "protein_weight": weight, "spectrum_weight": 1.0 - weight,
                "best_inner_balanced_accuracy": validation_score,
                "protein_c": c_value, "decision_threshold": 0.5,
            })
            payload = {
                "stage": 16, "model": "late_fusion", "seed": seed,
                "outer_fold": int(outer_fold), "protein_weight": weight,
                "spectrum_weight": 1.0 - weight, "protein_c": c_value,
                "selection_sample_ids": validation_ids, "test_sample_ids": test_ids,
            }
            (checkpoint_directory / f"late_fusion_seed_{seed}_fold_{outer_fold}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            progress["runtime"].append({
                "model": "late_fusion", "seed": seed, "outer_fold": int(outer_fold),
                "runtime_seconds": time.perf_counter() - started,
                "process_peak_memory_mib": peak_memory_mib(),
            })
            save_progress(table_directory, progress)


def run_trainable_hybrids(
    config: dict[str, Any], manifest: pd.DataFrame, protein_matrix: pd.DataFrame,
    bags: pd.DataFrame, metadata: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame,
    spectra_path: Path, reporter_path: Path, array_store: SpectrumArrayStore,
    peptide_to_id: dict[str, int], vocabulary_size: int,
    stage15_tuning: pd.DataFrame, checkpoint_directory: Path,
    table_directory: Path, progress: dict[str, list[dict[str, Any]]], device: torch.device,
) -> None:
    logger = logging.getLogger("single_cell_proteomics_mil")
    training = config["training"]
    downsample = int(config["spectrum"]["spectrum_downsample_factor"])
    spectrum_dim = array_store.spectrum_features.shape[1]
    label_by_sample = manifest.set_index("sample_id")["label_raw"]
    y_by_sample = label_by_sample.eq(config["positive_label"]).astype(int)
    inner_fold = int(config["validation_inner_fold"])
    for outer_fold in sorted(outer["outer_fold"].unique()):
        fold = outer[outer["outer_fold"].eq(outer_fold)]
        outer_train = fold.loc[fold["partition"].eq("train"), "sample_id"].tolist()
        test_ids = fold.loc[fold["partition"].eq("test"), "sample_id"].tolist()
        selected = inner[
            inner["outer_fold"].eq(outer_fold) & inner["inner_fold"].eq(inner_fold)
        ]
        selection_ids = selected.loc[selected["partition"].eq("train"), "sample_id"].tolist()
        validation_ids = selected.loc[selected["partition"].eq("validation"), "sample_id"].tolist()
        c_value = float(stage15_tuning[
            stage15_tuning["model"].eq("protein_logistic")
            & stage15_tuning["outer_fold"].eq(outer_fold)
        ]["selected_c"].iloc[0])
        for variant_index, variant in enumerate(training["variants"]):
            for seed in (int(value) for value in config["neural_seeds"]):
                if complete(progress, variant, seed, int(outer_fold), len(test_ids)):
                    continue
                started = time.perf_counter()
                run_seed = seed + int(outer_fold) * 1_000 + variant_index * 10_000
                logger.info("Stage 16 | fold=%s seed=%s model=%s", outer_fold, seed, variant)
                seed_everything(run_seed)
                protein_preprocessor = fit_protein_preprocessor(
                    protein_matrix, selection_ids,
                    float(config["protein"]["minimum_feature_prevalence"]),
                )
                transformed_selection = protein_preprocessor.transform(
                    protein_matrix.reindex(columns=selection_ids).T.to_numpy(float)
                )
                baseline = fit_residual_baseline(
                    transformed_selection, y_by_sample.loc[selection_ids].to_numpy(int),
                    c_value, run_seed,
                ) if variant == "residual_hybrid" else None
                spectrum_standardizer = fit_numeric_standardizer(
                    bags, metadata, selection_ids, spectra_path, reporter_path, True, array_store
                )
                selection_dataset = build_multimodal_dataset(
                    bags, metadata, selection_ids, spectra_path, reporter_path,
                    spectrum_standardizer, peptide_to_id,
                    int(training["maximum_training_instances_per_bag"]), run_seed,
                    array_store, protein_matrix, protein_preprocessor, downsample,
                )
                validation_dataset = build_multimodal_dataset(
                    bags, metadata, validation_ids, spectra_path, reporter_path,
                    spectrum_standardizer, peptide_to_id,
                    int(training["maximum_evaluation_instances_per_bag"]), run_seed,
                    array_store, protein_matrix, protein_preprocessor, downsample,
                )
                model = build_hybrid_model(
                    variant, transformed_selection.shape[1], spectrum_dim,
                    vocabulary_size, config, baseline,
                ).to(device)
                best_epoch, best_score, history, best_state = select_epochs(
                    model, selection_dataset, validation_dataset, bags, selection_ids,
                    training, device, variant, seed, int(outer_fold),
                )
                progress["history"].extend(history)
                selection_payload = {
                    "stage": 16, "model": variant, "seed": seed,
                    "training_seed": run_seed, "outer_fold": int(outer_fold),
                    "epochs": best_epoch, "best_inner_balanced_accuracy": best_score,
                    "protein_preprocessor": protein_preprocessor.as_dict(),
                    "spectrum_standardizer": spectrum_standardizer.as_dict(),
                    "training_sample_ids": selection_ids,
                    "validation_sample_ids": validation_ids,
                    "model_state_dict": best_state,
                }
                torch.save(
                    selection_payload,
                    checkpoint_directory / f"{variant}_seed_{seed}_fold_{outer_fold}_selection.pt",
                )
                selection_dataset.close(); validation_dataset.close()
                seed_everything(run_seed)
                final_preprocessor = fit_protein_preprocessor(
                    protein_matrix, outer_train,
                    float(config["protein"]["minimum_feature_prevalence"]),
                )
                transformed_outer = final_preprocessor.transform(
                    protein_matrix.reindex(columns=outer_train).T.to_numpy(float)
                )
                final_baseline = fit_residual_baseline(
                    transformed_outer, y_by_sample.loc[outer_train].to_numpy(int),
                    c_value, run_seed,
                ) if variant == "residual_hybrid" else None
                final_spectrum_standardizer = fit_numeric_standardizer(
                    bags, metadata, outer_train, spectra_path, reporter_path, True, array_store
                )
                final_dataset = build_multimodal_dataset(
                    bags, metadata, outer_train, spectra_path, reporter_path,
                    final_spectrum_standardizer, peptide_to_id,
                    int(training["maximum_training_instances_per_bag"]), run_seed,
                    array_store, protein_matrix, final_preprocessor, downsample,
                )
                final_model = build_hybrid_model(
                    variant, transformed_outer.shape[1], spectrum_dim,
                    vocabulary_size, config, final_baseline,
                ).to(device)
                progress["history"].extend(fit_final(
                    final_model, final_dataset, bags, outer_train, best_epoch,
                    training, device, variant, seed, int(outer_fold),
                ))
                final_dataset.close()
                test_dataset = build_multimodal_dataset(
                    bags, metadata, test_ids, spectra_path, reporter_path,
                    final_spectrum_standardizer, peptide_to_id,
                    int(training["maximum_evaluation_instances_per_bag"]), run_seed,
                    array_store, protein_matrix, final_preprocessor, downsample,
                )
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=positive_class_weight(bags, outer_train, device)
                )
                test_dataset.set_epoch(10_000)
                first_loader = make_loader(
                    test_dataset, int(training["evaluation_batch_size"]), False, run_seed
                )
                first_batch = next(iter(first_loader))
                invariance_error = permutation_invariance_error(final_model, first_batch, device)
                result = evaluate_repeated(
                    final_model, test_dataset, criterion, device,
                    int(training["evaluation_batch_size"]), run_seed,
                    int(training["evaluation_repeats"]),
                )
                test_dataset.close()
                final_payload = {
                    "stage": 16, "model": variant, "seed": seed,
                    "training_seed": run_seed, "outer_fold": int(outer_fold),
                    "epochs": best_epoch, "protein_preprocessor": final_preprocessor.as_dict(),
                    "spectrum_standardizer": final_spectrum_standardizer.as_dict(),
                    "training_sample_ids": outer_train, "test_sample_ids": test_ids,
                    "model_state_dict": final_model.state_dict(),
                }
                torch.save(
                    final_payload,
                    checkpoint_directory / f"{variant}_seed_{seed}_fold_{outer_fold}_final.pt",
                )
                add_prediction_rows(
                    progress["predictions"], variant, seed, int(outer_fold),
                    result["sample_ids"], label_by_sample, result["y_true"],
                    result["y_pred"], result["probabilities"],
                )
                progress["fold_metrics"].append({
                    "model": variant, "model_label": MODEL_LABELS[variant],
                    "seed": seed, "outer_fold": int(outer_fold),
                    "test_samples": len(test_ids), "selected_epochs": best_epoch,
                    "best_inner_balanced_accuracy": best_score,
                    "permutation_invariance_max_abs_error": invariance_error,
                    **result["metrics"],
                })
                gate = float(final_model.gate_value.detach().cpu()) if isinstance(
                    final_model, ResidualMultimodalMIL
                ) else np.nan
                progress["tuning"].append({
                    "model": variant, "seed": seed, "outer_fold": int(outer_fold),
                    "selected_epochs": best_epoch,
                    "best_inner_balanced_accuracy": best_score,
                    "protein_c": c_value, "residual_gate": gate,
                    "decision_threshold": 0.5,
                })
                progress["runtime"].append({
                    "model": variant, "seed": seed, "outer_fold": int(outer_fold),
                    "runtime_seconds": time.perf_counter() - started,
                    "process_peak_memory_mib": peak_memory_mib(),
                })
                save_progress(table_directory, progress)


def average_predictions(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    averaged = frame.groupby(
        ["model", "model_label", "sample_id", "label_raw", "y_true", "outer_fold"],
        as_index=False,
    )["probability_sc_u"].mean()
    averaged["y_pred"] = averaged["probability_sc_u"].ge(0.5).astype(int)
    metadata = manifest[[
        "sample_id", "raw_file_stem", "source_cohort", "batch_chromatography",
        "batch_sort", "batch_digest", "reporter_channel",
    ]].drop_duplicates("sample_id")
    return averaged.merge(metadata, on="sample_id", how="left", validate="many_to_one")


def build_reference_predictions(
    stage15: pd.DataFrame, manifest: pd.DataFrame,
) -> pd.DataFrame:
    selected = stage15[stage15["model"].isin(["protein_logistic", "spectrum_psm"])].copy()
    selected["model_label"] = selected["model"].map(MODEL_LABELS)
    return average_predictions(selected, manifest)


def evaluate_stage16(
    config: dict[str, Any], predictions: pd.DataFrame,
    stage15_predictions: pd.DataFrame, manifest: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    averaged_hybrids = average_predictions(predictions, manifest)
    references = build_reference_predictions(stage15_predictions, manifest)
    combined = pd.concat([references, averaged_hybrids], ignore_index=True)
    evaluation = config["evaluation"]
    model_rows, intervals, replicate_frames = [], [], []
    paired_rows, paired_frames = [], []
    calibration_frames = []
    for model_index, (model, frame) in enumerate(combined.groupby("model", sort=True)):
        metrics = binary_metrics(
            frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int),
            frame["probability_sc_u"].to_numpy(float),
        )
        calibration = calibration_metrics(
            frame["y_true"].to_numpy(int), frame["probability_sc_u"].to_numpy(float),
            int(evaluation["calibration_bins"]),
        )
        model_rows.append({
            "model": model, "model_label": MODEL_LABELS[model], "samples": len(frame),
            **metrics, **calibration,
        })
        bootstrap = group_bootstrap_metrics(
            frame, str(evaluation["bootstrap_group_column"]),
            int(evaluation["bootstrap_replicates"]), int(config["seed"]) + model_index * 10_000,
        )
        bootstrap.insert(0, "model", model)
        replicate_frames.append(bootstrap)
        for metric in DEFAULT_METRICS:
            lower, upper = percentile_interval(
                bootstrap[metric], float(evaluation["confidence_level"])
            )
            intervals.append({
                "model": model, "model_label": MODEL_LABELS[model], "metric": metric,
                "point_estimate": metrics[metric], "ci_lower": lower, "ci_upper": upper,
                "confidence_level": float(evaluation["confidence_level"]),
                "bootstrap_replicates": int(evaluation["bootstrap_replicates"]),
            })
        reliability = reliability_table(
            frame["y_true"].to_numpy(int), frame["probability_sc_u"].to_numpy(float),
            int(evaluation["calibration_bins"]),
        )
        reliability.insert(0, "model", model)
        calibration_frames.append(reliability)
    for hybrid_index, hybrid in enumerate(HYBRID_MODELS):
        comparator = combined[combined["model"].eq(hybrid)]
        comparator_metrics = binary_metrics(
            comparator["y_true"].to_numpy(int), comparator["y_pred"].to_numpy(int),
            comparator["probability_sc_u"].to_numpy(float),
        )
        for reference_index, reference_name in enumerate(("protein_logistic", "spectrum_psm")):
            reference = combined[combined["model"].eq(reference_name)]
            reference_metrics = binary_metrics(
                reference["y_true"].to_numpy(int), reference["y_pred"].to_numpy(int),
                reference["probability_sc_u"].to_numpy(float),
            )
            paired = paired_group_bootstrap_differences(
                reference, comparator, str(evaluation["bootstrap_group_column"]),
                int(evaluation["bootstrap_replicates"]),
                int(config["seed"]) + 100_000 + hybrid_index * 20_000 + reference_index * 10_000,
            )
            paired.insert(0, "comparator_model", hybrid)
            paired.insert(0, "reference_model", reference_name)
            paired_frames.append(paired)
            for metric in DEFAULT_METRICS:
                lower, upper = percentile_interval(
                    paired[metric], float(evaluation["confidence_level"])
                )
                paired_rows.append({
                    "reference_model": reference_name, "comparator_model": hybrid,
                    "metric": metric,
                    "point_difference_comparator_minus_reference": comparator_metrics[metric] - reference_metrics[metric],
                    "ci_lower": lower, "ci_upper": upper,
                    "confidence_level": float(evaluation["confidence_level"]),
                    "bootstrap_replicates": int(evaluation["bootstrap_replicates"]),
                })
    subgroup_rows = []
    for model, model_frame in combined.groupby("model"):
        for batch, frame in model_frame.groupby("batch_chromatography"):
            truth = frame["y_true"].to_numpy(int)
            row: dict[str, Any] = {
                "model": model, "model_label": MODEL_LABELS[model],
                "batch_chromatography": batch, "samples": len(frame),
                "negative_samples": int((truth == 0).sum()),
                "positive_samples": int((truth == 1).sum()),
            }
            if len(frame) >= int(evaluation["minimum_subgroup_samples"]) and np.unique(truth).size == 2:
                row.update(binary_metrics(
                    truth, frame["y_pred"].to_numpy(int),
                    frame["probability_sc_u"].to_numpy(float),
                ))
            else:
                row.update({metric: np.nan for metric in METRIC_COLUMNS})
            subgroup_rows.append(row)
    model_summary = pd.DataFrame(model_rows)
    interval_frame = pd.DataFrame(intervals)
    ba = interval_frame[interval_frame["metric"].eq("balanced_accuracy")][
        ["model", "ci_lower", "ci_upper"]
    ].rename(columns={"ci_lower": "balanced_accuracy_ci_lower", "ci_upper": "balanced_accuracy_ci_upper"})
    subgroup = pd.DataFrame(subgroup_rows)
    worst = subgroup.groupby("model")["balanced_accuracy"].min().rename("worst_batch_balanced_accuracy")
    model_summary = model_summary.merge(ba, on="model", how="left")
    model_summary["worst_batch_balanced_accuracy"] = model_summary["model"].map(worst)
    model_summary = model_summary.sort_values("balanced_accuracy", ascending=False)
    return {
        "averaged_predictions": combined,
        "model_summary": model_summary,
        "bootstrap_intervals": interval_frame,
        "bootstrap_replicates": pd.concat(replicate_frames, ignore_index=True),
        "paired_comparisons": pd.DataFrame(paired_rows),
        "paired_replicates": pd.concat(paired_frames, ignore_index=True),
        "calibration": pd.concat(calibration_frames, ignore_index=True),
        "subgroup_metrics": subgroup,
    }


def complementarity_analysis(averaged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = averaged.pivot(index="sample_id", columns="model", values=["y_true", "y_pred", "probability_sc_u"])
    truth = values[("y_true", "protein_logistic")].astype(int)
    per_cell = pd.DataFrame({"sample_id": values.index, "y_true": truth.to_numpy()})
    for model in MODEL_LABELS:
        per_cell[f"{model}_probability"] = values[("probability_sc_u", model)].to_numpy(float)
        per_cell[f"{model}_correct"] = values[("y_pred", model)].astype(int).to_numpy() == truth.to_numpy()
    protein_correct = per_cell["protein_logistic_correct"]
    spectrum_correct = per_cell["spectrum_psm_correct"]
    per_cell["base_error_category"] = np.select(
        [protein_correct & spectrum_correct, protein_correct & ~spectrum_correct,
         ~protein_correct & spectrum_correct],
        ["both_correct", "protein_only_correct", "spectrum_only_correct"],
        default="both_wrong",
    )
    rows = []
    for category, frame in per_cell.groupby("base_error_category"):
        row: dict[str, Any] = {"base_error_category": category, "cells": len(frame)}
        for hybrid in HYBRID_MODELS:
            row[f"{hybrid}_correct_cells"] = int(frame[f"{hybrid}_correct"].sum())
            row[f"{hybrid}_accuracy"] = float(frame[f"{hybrid}_correct"].mean())
        rows.append(row)
    oracle = protein_correct | spectrum_correct
    rows.append({
        "base_error_category": "protein_or_spectrum_oracle", "cells": len(per_cell),
        **{f"{hybrid}_correct_cells": int(per_cell[f"{hybrid}_correct"].sum()) for hybrid in HYBRID_MODELS},
        **{f"{hybrid}_accuracy": float(per_cell[f"{hybrid}_correct"].mean()) for hybrid in HYBRID_MODELS},
        "oracle_correct_cells": int(oracle.sum()), "oracle_accuracy": float(oracle.mean()),
    })
    return per_cell, pd.DataFrame(rows)


def make_figures(
    model_summary: pd.DataFrame, tuning: pd.DataFrame,
    complementarity: pd.DataFrame, averaged: pd.DataFrame, output: Path,
) -> None:
    ordered = model_summary.sort_values("balanced_accuracy")
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x = ordered["balanced_accuracy"].to_numpy(float)
    axis.errorbar(
        x, ordered["model_label"],
        xerr=np.vstack((x - ordered["balanced_accuracy_ci_lower"], ordered["balanced_accuracy_ci_upper"] - x)),
        fmt="o", capsize=3,
    )
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set(xlim=(0, 1), xlabel="Balanced accuracy with 95% raw-file bootstrap CI", title="Stage 16 protein and spectrum fusion")
    figure.savefig(output / "stage_16_model_comparison.png", dpi=180); plt.close(figure)

    weights = tuning[tuning["model"].eq("late_fusion")]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for seed, frame in weights.groupby("seed"):
        axis.plot(frame["outer_fold"], frame["protein_weight"], marker="o", label=f"Seed {seed}")
    axis.set(ylim=(-0.03, 1.03), xlabel="Outer fold", ylabel="Validation-selected protein weight", title="Late-fusion weights")
    axis.legend()
    figure.savefig(output / "stage_16_fusion_weights.png", dpi=180); plt.close(figure)

    categories = complementarity[~complementarity["base_error_category"].eq("protein_or_spectrum_oracle")]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(categories["base_error_category"], categories["cells"])
    axis.set_ylabel("Cells")
    axis.set_title("Protein and spectrum prediction overlap")
    axis.tick_params(axis="x", rotation=20)
    figure.savefig(output / "stage_16_error_overlap.png", dpi=180); plt.close(figure)

    protein = averaged[averaged["model"].eq("protein_logistic")].set_index("sample_id")
    spectrum = averaged[averaged["model"].eq("spectrum_psm")].set_index("sample_id").reindex(protein.index)
    figure, axis = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    colors = np.where(protein["y_true"].to_numpy(int) == 1, "#d95f02", "#1b9e77")
    axis.scatter(protein["probability_sc_u"], spectrum["probability_sc_u"], c=colors, s=16, alpha=0.65)
    axis.axvline(0.5, color="gray", linewidth=1, linestyle="--"); axis.axhline(0.5, color="gray", linewidth=1, linestyle="--")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Protein probability of sc_u", ylabel="Spectrum probability of sc_u", title="Protein-spectrum prediction complementarity")
    figure.savefig(output / "stage_16_probability_complementarity.png", dpi=180); plt.close(figure)


def validate_outputs(
    predictions: pd.DataFrame, fold_metrics: pd.DataFrame,
    expected_ids: set[str], seeds: list[int],
) -> None:
    if set(predictions["model"]) != set(HYBRID_MODELS):
        raise ValueError("Stage 16 predictions do not contain all three hybrids")
    for (model, seed), frame in predictions.groupby(["model", "seed"]):
        if len(frame) != len(expected_ids) or set(frame["sample_id"]) != expected_ids:
            raise ValueError(f"{model} seed {seed} lacks exactly one prediction per cell")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{model} seed {seed} contains duplicate cell predictions")
    if set(predictions["seed"].astype(int)) != set(seeds):
        raise ValueError("Stage 16 neural seed coverage is incomplete")
    if fold_metrics["permutation_invariance_max_abs_error"].max() > 1e-5:
        raise ValueError("a multimodal model is not spectrum-instance permutation invariant")


def render_report(summary: dict[str, Any], model_summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    result_lines = [
        f"- `{row.model_label}`: balanced accuracy {row.balanced_accuracy:.3f} "
        f"(95% CI {row.balanced_accuracy_ci_lower:.3f}–{row.balanced_accuracy_ci_upper:.3f}); "
        f"macro F1 {row.macro_f1:.3f}; AUROC {row.auroc:.3f}."
        for row in model_summary.itertuples(index=False)
    ]
    protein_differences = paired[
        paired["reference_model"].eq("protein_logistic")
        & paired["metric"].eq("balanced_accuracy")
    ]
    difference_lines = [
        f"- `{MODEL_LABELS[row.comparator_model]}` minus protein logistic: {row.point_difference_comparator_minus_reference:+.3f} "
        f"(95% paired CI {row.ci_lower:+.3f} to {row.ci_upper:+.3f})."
        for row in protein_differences.itertuples(index=False)
    ]
    return "\n".join([
        "# Stage 16 — Multimodal protein-plus-spectrum MIL", "",
        "## Design", "",
        f"- Cells: {summary['cells']:,} across {summary['raw_file_groups']:,} raw-file groups.",
        "- Protein and spectrum inputs use the same Stage 14 cells and Stage 14 grouped outer/inner splits.",
        "- Late fusion selects its protein-versus-spectrum weight using inner validation predictions only.",
        "- Joint fusion concatenates fold-preprocessed protein and gated-attention spectrum-bag embeddings.",
        "- Residual fusion starts from a frozen training-fold protein logistic prediction; its zero-initialized gated spectrum residual must learn incremental validation value.",
        f"- Trainable hybrids use deterministic seeds {', '.join(map(str, summary['neural_seeds']))}.",
        "- No raw-file name, TMT-set identifier, or batch identifier is supplied as a predictive feature.", "",
        "## Results", "", *result_lines, "",
        "## Incremental value over protein abundance", "", *difference_lines, "",
        f"- Best hybrid: `{summary['best_hybrid_label']}` at {summary['best_hybrid_balanced_accuracy']:.3f} balanced accuracy.",
        f"- Best hybrid improvement over protein logistic: {summary['best_hybrid_difference']:+.3f} balanced-accuracy points.",
        "- A confidence interval spanning zero is interpreted as no reproducible improvement.", "",
        "## Integrity and resources", "",
        f"- Saved {summary['prediction_rows']:,} seed-specific hybrid OOF predictions and {summary['checkpoint_files']:,} Stage 16 checkpoints.",
        f"- Maximum observed within-bag permutation error: {summary['maximum_permutation_error']:.2e}.",
        f"- Runtime: {summary['runtime_minutes']:.1f} minutes on `{summary['device']}`; peak process memory {summary['peak_memory_mib']:.1f} MiB.",
        "- Stage 16 passes because every hybrid has one OOF prediction per cell and seed, all fusion choices are outer-test independent, inputs remain aligned, and spectrum-instance order does not affect predictions beyond numerical tolerance.", "",
    ])


def run_stage_sixteen(
    config_path: str | Path = "configs/stage_16.yaml", dry_run: bool = False,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    data = config["data"]
    directory = Path(data["directory"])
    manifest = pd.read_parquet(data["manifest"])
    protein = pd.read_parquet(data["protein_matrix"])
    bags = pd.read_parquet(directory / "spectrum_bag_index.parquet")
    metadata = pd.read_parquet(directory / "spectrum_metadata.parquet")
    spectra_path = directory / "spectra.h5"
    reporter_path = directory / "reporter_intensities.h5"
    outer = pd.read_csv(data["outer_splits"])
    inner = pd.read_csv(data["inner_splits"])
    stage15_predictions = pd.read_csv(data["stage_15_predictions"])
    stage15_tuning = pd.read_csv(data["stage_15_tuning"])
    ids = set(manifest["sample_id"].astype(str))
    if ids != set(protein.columns.astype(str)) or ids != set(bags["sample_id"].astype(str)):
        raise ValueError("Stage 16 protein, spectrum, and manifest cells are not aligned")
    validate_spectrum_bags(bags, metadata, spectra_path, reporter_path)
    peptide_to_id, vocabulary = make_psm_vocabulary(metadata)
    vocabulary_size = int(vocabulary["peptide_id"].max()) + 1
    array_store = SpectrumArrayStore.from_hdf5(
        spectra_path, reporter_path, int(config["spectrum"]["spectrum_downsample_factor"])
    )
    if dry_run:
        preprocessor = fit_protein_preprocessor(
            protein, manifest["sample_id"].head(50).tolist(),
            float(config["protein"]["minimum_feature_prevalence"]),
        )
        return {
            "dry_run": True, "cells": len(manifest), "protein_features": len(protein),
            "retained_test_protein_features": int(preprocessor.support.sum()),
            "spectrum_dimensions": int(array_store.spectrum_features.shape[1]),
            "psm_vocabulary_size": vocabulary_size, "hybrid_models": list(HYBRID_MODELS),
        }
    checkpoint_directory = Path("outputs/checkpoints/stage_16")
    table_directory = Path("outputs/tables")
    figure_directory = Path("outputs/figures")
    report_directory = Path("outputs/reports")
    for path in (checkpoint_directory, table_directory, figure_directory, report_directory):
        path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config["training"]["device"]))
    started = time.perf_counter()
    progress = load_progress(table_directory, bool(config["runtime"].get("resume", True)))
    run_late_fusion(
        config, manifest, protein, bags, metadata, outer, spectra_path, reporter_path,
        array_store, peptide_to_id, vocabulary_size, stage15_predictions, stage15_tuning,
        checkpoint_directory, table_directory, progress, device,
    )
    run_trainable_hybrids(
        config, manifest, protein, bags, metadata, outer, inner, spectra_path, reporter_path,
        array_store, peptide_to_id, vocabulary_size, stage15_tuning,
        checkpoint_directory, table_directory, progress, device,
    )
    predictions = pd.DataFrame(progress["predictions"])
    fold_metrics = pd.DataFrame(progress["fold_metrics"])
    history = pd.DataFrame(progress["history"])
    runtime = pd.DataFrame(progress["runtime"])
    tuning = pd.DataFrame(progress["tuning"])
    validate_outputs(
        predictions, fold_metrics, ids, [int(value) for value in config["neural_seeds"]]
    )
    evaluated = evaluate_stage16(config, predictions, stage15_predictions, manifest)
    error_cells, error_summary = complementarity_analysis(evaluated["averaged_predictions"])
    predictions.to_csv(table_directory / "stage_16_predictions.csv", index=False)
    fold_metrics.to_csv(table_directory / "stage_16_fold_seed_metrics.csv", index=False)
    evaluated["model_summary"].to_csv(table_directory / "stage_16_model_summary.csv", index=False)
    tuning.to_csv(table_directory / "stage_16_fusion_tuning.csv", index=False)
    evaluated["paired_comparisons"].to_csv(table_directory / "stage_16_paired_comparisons.csv", index=False)
    evaluated["bootstrap_intervals"].to_csv(table_directory / "stage_16_bootstrap_intervals.csv", index=False)
    evaluated["calibration"].to_csv(table_directory / "stage_16_calibration.csv", index=False)
    evaluated["subgroup_metrics"].to_csv(table_directory / "stage_16_subgroup_metrics.csv", index=False)
    history.to_csv(table_directory / "stage_16_training_history.csv", index=False)
    runtime.to_csv(table_directory / "stage_16_runtime_memory.csv", index=False)
    error_summary.to_csv(table_directory / "stage_16_error_overlap_summary.csv", index=False)
    evaluated["averaged_predictions"].to_parquet(
        table_directory / "stage_16_averaged_predictions.parquet", index=False
    )
    evaluated["bootstrap_replicates"].to_parquet(
        table_directory / "stage_16_bootstrap_replicates.parquet", index=False
    )
    evaluated["paired_replicates"].to_parquet(
        table_directory / "stage_16_paired_bootstrap_replicates.parquet", index=False
    )
    error_cells.to_parquet(table_directory / "stage_16_error_overlap_by_cell.parquet", index=False)
    make_figures(
        evaluated["model_summary"], tuning, error_summary,
        evaluated["averaged_predictions"], figure_directory,
    )
    hybrid_summary = evaluated["model_summary"][
        evaluated["model_summary"]["model"].isin(HYBRID_MODELS)
    ].sort_values("balanced_accuracy", ascending=False)
    best = hybrid_summary.iloc[0]
    protein_ba = float(evaluated["model_summary"].set_index("model").loc["protein_logistic", "balanced_accuracy"])
    elapsed = time.perf_counter() - started
    summary = {
        "cells": len(manifest), "raw_file_groups": int(manifest["raw_file_stem"].nunique()),
        "neural_seeds": [int(value) for value in config["neural_seeds"]],
        "prediction_rows": len(predictions),
        "checkpoint_files": len(list(checkpoint_directory.glob("*"))),
        "best_hybrid": str(best["model"]), "best_hybrid_label": str(best["model_label"]),
        "best_hybrid_balanced_accuracy": float(best["balanced_accuracy"]),
        "best_hybrid_difference": float(best["balanced_accuracy"] - protein_ba),
        "maximum_permutation_error": float(fold_metrics["permutation_invariance_max_abs_error"].max()),
        "runtime_seconds": elapsed, "runtime_minutes": elapsed / 60,
        "peak_memory_mib": peak_memory_mib(), "device": str(device),
    }
    (report_directory / "stage_16_multimodal_protein_spectrum_mil.md").write_text(
        render_report(summary, evaluated["model_summary"], evaluated["paired_comparisons"]),
        encoding="utf-8",
    )
    (table_directory / "stage_16_run_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for name in PROGRESS_NAMES:
        path = table_directory / f"stage_16_{name}.partial.csv"
        if path.exists():
            path.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage_16.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_16_multimodal_mil.log"))
    result = run_stage_sixteen(args.config, args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 16 completed: %s", result)


if __name__ == "__main__":
    main()
