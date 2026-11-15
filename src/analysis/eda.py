"""Stage 4 exploratory analysis and technical-confounding audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.utils.logging import configure_logging


def prepare_pca(peptide_matrix: pd.DataFrame) -> pd.DataFrame:
    """Create two-component PCA coordinates from log-transformed peptide features."""
    values = np.log1p(peptide_matrix.fillna(0).T)
    coordinates = PCA(n_components=2, random_state=2026).fit_transform(StandardScaler().fit_transform(values))
    return pd.DataFrame(coordinates, index=values.index, columns=["PC1", "PC2"])


def batch_only_diagnostic(metadata: pd.DataFrame) -> dict[str, float | int]:
    """Estimate how well technical batches alone predict labels; a confounding diagnostic."""
    features = pd.get_dummies(metadata[["batch_chromatography", "batch_sort", "batch_digest"]].astype(str), dtype=float)
    labels = metadata["label_raw"].to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    predicted = cross_val_predict(model, features, labels, cv=folds)
    return {"samples": len(metadata), "features": features.shape[1], "random_cv_balanced_accuracy": float(balanced_accuracy_score(labels, predicted))}


def create_figure(metadata: pd.DataFrame, pca: pd.DataFrame, output_path: Path) -> None:
    """Render PCA and detection-QC panels for the Stage 4 report."""
    frame = metadata.join(pca, on="sample_id")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for label, group in frame.groupby("label_raw"):
        axes[0].scatter(group["PC1"], group["PC2"], label=label, alpha=0.75, s=22)
    axes[0].set(title="PCA by cell label", xlabel="PC1", ylabel="PC2")
    axes[0].legend(title="Label")
    for batch, group in frame.groupby("batch_chromatography"):
        axes[1].scatter(group["PC1"], group["PC2"], label=batch, alpha=0.75, s=22)
    axes[1].set(title="PCA by chromatography batch", xlabel="PC1", ylabel="PC2")
    axes[1].legend(title="Batch")
    order = sorted(frame["batch_chromatography"].unique())
    data = [frame.loc[frame["batch_chromatography"].eq(batch), "detected_peptides"] for batch in order]
    axes[2].boxplot(data, tick_labels=order, showfliers=False)
    axes[2].set(title="Detected peptides by batch", xlabel="Chromatography batch", ylabel="Count")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_stage_four() -> dict[str, float | int]:
    """Run the Stage 4 audit and write tables, plots, and Markdown findings."""
    manifest = pd.read_parquet("data/interim/manifests/master_manifest.parquet")
    peptide = pd.read_parquet("data/processed/fixed_features/peptide_by_sample.parquet")
    qc = pd.read_csv("outputs/tables/stage_03_sample_qc.csv")
    metadata = manifest.loc[manifest["is_valid_target"], ["sample_id", "label_raw", "raw_file_stem", "batch_chromatography", "batch_sort", "batch_digest", "reporter_channel"]].merge(qc[["sample_id", "detected_peptides", "detected_proteins", "peptide_missing_fraction", "total_peptide_reporter_intensity"]], on="sample_id", validate="one_to_one")
    pca = prepare_pca(peptide)
    duplicate_mask = peptide.fillna(0).T.duplicated(keep=False)
    duplicate_samples = pca.index[duplicate_mask].tolist()
    diagnostics = batch_only_diagnostic(metadata)
    tables, figures, reports = Path("outputs/tables"), Path("outputs/figures"), Path("outputs/reports")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    metadata.join(pca, on="sample_id").to_csv(tables / "stage_04_pca_coordinates.csv", index=False)
    metadata.groupby(["label_raw", "raw_file_stem", "batch_chromatography", "batch_sort", "batch_digest"], dropna=False).size().rename("sample_count").reset_index().to_csv(tables / "stage_04_class_batch_counts.csv", index=False)
    pd.DataFrame({"sample_id": duplicate_samples}).to_csv(tables / "stage_04_exact_duplicate_samples.csv", index=False)
    (tables / "stage_04_batch_only_diagnostic.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    create_figure(metadata, pca, figures / "stage_04_eda.png")
    (reports / "stage_04_eda_leakage_audit.md").write_text("\n".join([
        "# Stage 4 — Exploratory analysis and leakage audit", "",
        "## Dataset composition", "", f"- Usable samples: {len(metadata)}", f"- Raw-file groups: {metadata['raw_file_stem'].nunique()}", f"- Chromatography batches: {metadata['batch_chromatography'].nunique()}", f"- Sorting days: {metadata['batch_sort'].nunique()}", f"- Digestion batches: {metadata['batch_digest'].nunique()}", f"- Labels: {metadata['label_raw'].value_counts().to_dict()}", "",
        "## Technical-confounding diagnostic", "", f"- Random-CV balanced accuracy from technical batch fields alone: {diagnostics['random_cv_balanced_accuracy']:.3f}", "- This diagnostic is not a biological performance estimate. It measures whether labels are predictable from technical batch composition and must be interpreted alongside group-aware evaluation.", "",
        "## Duplicate and QC checks", "", f"- Exact duplicate peptide profiles: {len(duplicate_samples)}", f"- Samples with zero detected peptides: {int(metadata['detected_peptides'].eq(0).sum())}", "- Missingness remains un-imputed; transformations must be fit on training samples only after split assignment.", "",
        "## Gate", "", "The dataset has usable peptide/protein features and multiple raw-file groups. Later modeling must group splits by raw file (or a stricter batch grouping) and must report batch-specific results.", "",
    ]), encoding="utf-8")
    return {**diagnostics, "duplicate_samples": len(duplicate_samples), "zero_peptide_samples": int(metadata["detected_peptides"].eq(0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logger = configure_logging(Path("outputs/logs/stage_04_eda.log"))
    result = run_stage_four()
    logger.info("Stage 4 completed: %s", result)


if __name__ == "__main__":
    main()
