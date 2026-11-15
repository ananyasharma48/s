"""Derive leakage-safe-ready peptide and protein abundance datasets from evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


REPORTER_FIELDS = [f"Reporter.intensity.corrected.{index}" for index in range(17)]


def is_flagged(values: pd.Series) -> pd.Series:
    """Identify MaxQuant-style reverse/contaminant flags without treating NA as true."""
    return values.fillna("").astype(str).str.strip().str.lower().isin({"+", "true", "1"})


def derive_long_features(evidence: pd.DataFrame, manifest: pd.DataFrame, max_pep: float) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand evidence rows into channel-specific peptide observations and apply QC."""
    required = {"Raw.file", "Modified.sequence", "Leading.proteins", "Reverse", "Potential.contaminant", "PEP", *REPORTER_FIELDS}
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"evidence is missing columns: {sorted(missing)}")
    valid = manifest.loc[manifest["is_valid_target"], ["sample_id", "raw_file_stem", "reporter_index"]].copy()
    targets = set(valid["raw_file_stem"])
    scoped = evidence[evidence["Raw.file"].astype(str).isin(targets)].copy()
    stats = {"input_rows": len(scoped)}
    if scoped.empty:
        return pd.DataFrame(columns=["peptide", "protein", "intensity", "sample_id"]), {**stats, "quality_retained_rows": 0, "positive_channel_observations": 0}
    quality = (~is_flagged(scoped["Reverse"]) & ~is_flagged(scoped["Potential.contaminant"]))
    pep = pd.to_numeric(scoped["PEP"], errors="coerce")
    quality &= pep.notna() & (pep <= max_pep)
    quality &= scoped["Modified.sequence"].notna() & scoped["Leading.proteins"].notna()
    scoped = scoped[quality]
    stats["quality_retained_rows"] = len(scoped)
    records = []
    for raw_file, sample_rows in valid.groupby("raw_file_stem"):
        run = scoped[scoped["Raw.file"].astype(str).eq(raw_file)]
        if run.empty:
            continue
        for sample in sample_rows.itertuples(index=False):
            field = f"Reporter.intensity.corrected.{sample.reporter_index}"
            intensity = pd.to_numeric(run[field], errors="coerce")
            selected = run.loc[intensity.gt(0), ["Modified.sequence", "Leading.proteins"]].copy()
            selected["intensity"] = intensity.loc[selected.index]
            selected["sample_id"] = sample.sample_id
            records.append(selected)
    if not records:
        return pd.DataFrame(columns=["peptide", "protein", "intensity", "sample_id"]), {**stats, "positive_channel_observations": 0}
    long = pd.concat(records, ignore_index=True).rename(columns={"Modified.sequence": "peptide", "Leading.proteins": "protein"})
    stats["positive_channel_observations"] = len(long)
    return long, stats


def matrix_from_long(long: pd.DataFrame, feature_column: str, sample_ids: list[str]) -> pd.DataFrame:
    """Aggregate reporter intensity into a feature-by-sample matrix, preserving zeros as missing."""
    grouped = long.groupby([feature_column, "sample_id"], observed=True)["intensity"].sum()
    matrix = grouped.unstack("sample_id").reindex(columns=sample_ids)
    return matrix.sort_index()


def sample_qc(peptide_matrix: pd.DataFrame, protein_matrix: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature detection and intensity by valid sample."""
    frame = manifest.loc[manifest["is_valid_target"], ["sample_id", "label_raw", "raw_file_stem", "batch_chromatography", "batch_sort", "batch_digest"]].set_index("sample_id")
    frame["detected_peptides"] = peptide_matrix.notna().sum(axis=0)
    frame["detected_proteins"] = protein_matrix.notna().sum(axis=0)
    frame["peptide_missing_fraction"] = peptide_matrix.isna().mean(axis=0)
    frame["protein_missing_fraction"] = protein_matrix.isna().mean(axis=0)
    frame["total_peptide_reporter_intensity"] = peptide_matrix.sum(axis=0, min_count=1)
    return frame.reset_index()


def run_stage_three(data_config_path: str | Path, preprocessing_config_path: str | Path) -> dict[str, int]:
    """Create Stage-3 feature matrices and QC outputs from evidence and the manifest."""
    data_config = load_yaml_config(data_config_path)
    preprocessing = load_yaml_config(preprocessing_config_path)
    manifest = pd.read_parquet("data/interim/manifests/master_manifest.parquet")
    evidence_path = Path(data_config["paths"]["evidence_csv"])
    usecols = ["Raw.file", "Modified.sequence", "Leading.proteins", "Reverse", "Potential.contaminant", "PEP", *REPORTER_FIELDS]
    chunks = []
    stats = {"input_rows": 0, "quality_retained_rows": 0, "positive_channel_observations": 0}
    for chunk in pd.read_csv(evidence_path, usecols=usecols, chunksize=100_000, low_memory=False):
        long, chunk_stats = derive_long_features(chunk, manifest, float(preprocessing["quality_control"]["max_pep"]))
        if not long.empty:
            chunks.append(long)
        for key, value in chunk_stats.items():
            stats[key] += value
    if not chunks:
        raise ValueError("no positive channel-specific reporter intensities remained after quality filtering")
    full_long = pd.concat(chunks, ignore_index=True)
    sample_ids = manifest.loc[manifest["is_valid_target"], "sample_id"].tolist()
    peptide_matrix = matrix_from_long(full_long, "peptide", sample_ids)
    protein_matrix = matrix_from_long(full_long, "protein", sample_ids)
    qc = sample_qc(peptide_matrix, protein_matrix, manifest)
    processed = Path("data/processed/fixed_features")
    tables = Path("outputs/tables")
    reports = Path("outputs/reports")
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    full_long.to_parquet(processed / "channel_peptide_evidence.parquet", index=False)
    peptide_matrix.to_parquet(processed / "peptide_by_sample.parquet")
    protein_matrix.to_parquet(processed / "protein_by_sample.parquet")
    qc.to_csv(tables / "stage_03_sample_qc.csv", index=False)
    qc.groupby(["label_raw", "batch_chromatography", "batch_sort", "batch_digest"], dropna=False).size().rename("sample_count").reset_index().to_csv(tables / "stage_03_class_batch_distribution.csv", index=False)
    (reports / "stage_03_processed_data_qc.md").write_text("\n".join([
        "# Stage 3 — Processed-data extraction and quality control", "",
        "## Scope", "", "Derived from the local LCA9/LCA10/LCB7 evidence rows and the Stage 2 valid-target manifest.", "",
        "## QC rules", "", "- Reverse and potential-contaminant evidence is excluded.", f"- PEP must be at most {preprocessing['quality_control']['max_pep']}.", "- Only positive channel-specific corrected reporter intensity is retained.", "- Missing matrix values represent non-detection; no imputation, scaling, or feature filtering has been fitted here.", "",
        "## Outputs", "", f"- Valid samples: {len(sample_ids)}", f"- Peptide features: {len(peptide_matrix)}", f"- Protein features: {len(protein_matrix)}", f"- Input evidence rows within local runs: {stats['input_rows']}", f"- Quality-retained evidence rows: {stats['quality_retained_rows']}", f"- Positive channel-specific observations: {stats['positive_channel_observations']}", "",
        "## Gate", "", "These are derived feature datasets. Training-only transforms, prevalence filtering, and imputation are deferred until leakage-safe split assignment.", "",
    ]), encoding="utf-8")
    return {**stats, "valid_samples": len(sample_ids), "peptide_features": len(peptide_matrix), "protein_features": len(protein_matrix)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    args = parser.parse_args()
    logger = configure_logging(Path("outputs/logs/stage_03_processed_qc.log"))
    stats = run_stage_three(args.data_config, args.preprocessing_config)
    logger.info("Stage 3 completed: %s", stats)


if __name__ == "__main__":
    main()
