"""Execute Stage 14 matched protein-vector and spectrum-bag reconstruction."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from src.data.build_spectrum_bags import (
    MAX_REPORTER_CHANNELS,
    _append_dataset,
    process_mzml_run,
    validate_spectrum_outputs,
)
from src.data.build_splits import (
    assert_group_disjoint,
    make_batch_holdouts,
    make_inner_assignments,
    make_outer_assignments,
)
from src.data.extract_reporter_ions import file_checksum, inspect_mzml_header, validate_reporter_scheme
from src.data.inventory import sha256sum
from src.data.preprocess_abundance import is_flagged
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


FULL_PSM_COLUMNS = [
    "Raw.file",
    "MS.MS.scan.number",
    "Sequence",
    "Modified.sequence",
    "Leading.razor.protein",
    "Leading.proteins",
    "Reverse",
    "Potential.contaminant",
    "dart_qval",
    "PEP",
    "PIF",
    "Score",
]


def load_full_psm_lookup(
    evidence_path: str | Path,
    raw_file_stems: set[str],
) -> pd.DataFrame:
    """Load the best search-result identity for each locally available MS2 scan."""
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        evidence_path,
        usecols=FULL_PSM_COLUMNS,
        chunksize=100_000,
        low_memory=False,
    ):
        selected = chunk[chunk["Raw.file"].astype(str).isin(raw_file_stems)].copy()
        if selected.empty:
            continue
        selected["MS.MS.scan.number"] = pd.to_numeric(
            selected["MS.MS.scan.number"], errors="coerce"
        )
        selected = selected.dropna(subset=["MS.MS.scan.number"])
        selected["MS.MS.scan.number"] = selected["MS.MS.scan.number"].astype(np.int64)
        for column in ["dart_qval", "PEP", "PIF", "Score"]:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
        frames.append(selected)
    if not frames:
        raise ValueError("no search-result rows match the Stage 14 acquisition set")
    lookup = pd.concat(frames, ignore_index=True)
    lookup = lookup.sort_values(
        ["Raw.file", "MS.MS.scan.number", "dart_qval", "PEP", "Score"],
        ascending=[True, True, True, True, False],
        na_position="last",
    ).drop_duplicates(["Raw.file", "MS.MS.scan.number"], keep="first")
    return lookup.rename(columns={
        "Raw.file": "raw_file_stem",
        "MS.MS.scan.number": "scan_id",
        "Sequence": "psm_sequence_full",
        "Modified.sequence": "psm_modified_sequence",
        "Leading.razor.protein": "psm_leading_razor_protein",
        "Leading.proteins": "psm_leading_proteins",
        "Reverse": "psm_reverse",
        "Potential.contaminant": "psm_potential_contaminant",
        "dart_qval": "psm_q_value_full",
        "PEP": "psm_pep_full",
        "PIF": "psm_precursor_ion_fraction_full",
        "Score": "psm_score_full",
    })


def basic_psm_lookup(full_lookup: pd.DataFrame) -> pd.DataFrame:
    """Adapt the full lookup to the columns consumed by spectrum preprocessing."""
    return full_lookup[[
        "raw_file_stem",
        "scan_id",
        "psm_sequence_full",
        "psm_leading_razor_protein",
        "psm_q_value_full",
        "psm_pep_full",
        "psm_precursor_ion_fraction_full",
        "psm_score_full",
    ]].rename(columns={
        "psm_sequence_full": "psm_sequence",
        "psm_leading_razor_protein": "psm_leading_protein",
        "psm_q_value_full": "psm_q_value",
        "psm_pep_full": "psm_pep",
        "psm_precursor_ion_fraction_full": "psm_precursor_ion_fraction",
        "psm_score_full": "psm_score",
    })


def append_novel_spectra(
    config: dict[str, Any],
    stage: dict[str, Any],
    manifest: pd.DataFrame,
    run_summary: pd.DataFrame,
    full_lookup: pd.DataFrame,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path]:
    """Copy Stage 11 stores and append the 76 deduplicated novel acquisitions."""
    output_directory = Path(stage["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    spectra_path = output_directory / "spectra.h5"
    reporter_path = output_directory / "reporter_intensities.h5"
    metadata_path = output_directory / "spectrum_metadata.parquet"
    novel = run_summary[run_summary["acquisition_status"].eq("ready_new_stage13")].copy()
    if novel.empty:
        raise ValueError("Stage 14 has no novel acquisitions to append")
    candidates_by_run = manifest[manifest["stage13_candidate"]].groupby("raw_file_stem")["sample_id"].nunique()
    novel["valid_target_cells"] = novel["raw_file_stem"].map(candidates_by_run).fillna(0).astype(int)
    for scheme in novel["reporter_scheme"].unique():
        masses = config["reporter_schemes"][scheme]["masses"]
        validate_reporter_scheme(
            {str(channel): float(mass) for channel, mass in masses.items()},
            float(stage["reporter_tolerance_da"]),
        )
    if dry_run:
        return novel, pd.DataFrame(), pd.DataFrame(), spectra_path, reporter_path

    source_directory = Path(stage["stage11_spectrum_directory"])
    source_spectra = source_directory / "spectra.h5"
    source_reporters = source_directory / "reporter_intensities.h5"
    source_metadata = source_directory / "spectrum_metadata.parquet"
    for path in [source_spectra, source_reporters, source_metadata]:
        if not path.is_file():
            raise FileNotFoundError(f"required Stage 11 artifact is missing: {path}")
    spectra_tmp = output_directory / "spectra.h5.tmp"
    reporter_tmp = output_directory / "reporter_intensities.h5.tmp"
    shutil.copy2(source_spectra, spectra_tmp)
    shutil.copy2(source_reporters, reporter_tmp)
    existing_metadata = pd.read_parquet(source_metadata)
    metadata_frames = [existing_metadata]
    new_qc_frames: list[pd.DataFrame] = []
    new_records: list[dict[str, Any]] = []
    representation = stage["representation"]
    logger = logging.getLogger("single_cell_proteomics_mil")
    psm_lookup = basic_psm_lookup(full_lookup)
    start_raw_id = int(existing_metadata["raw_file_id"].max()) + 1

    with h5py.File(spectra_tmp, "a") as spectra_store, h5py.File(reporter_tmp, "a") as reporter_store:
        embedding_ds = spectra_store["spectrum_embedding"]
        precursor_ds = spectra_store["precursor_features"]
        scan_ds = spectra_store["scan_id"]
        raw_id_ds = spectra_store["raw_file_id"]
        reporter_ds = reporter_store["reporter_intensities"]
        reporter_scan_ds = reporter_store["scan_id"]
        reporter_raw_id_ds = reporter_store["raw_file_id"]
        for offset, row in enumerate(novel.sort_values("raw_file_stem").itertuples(index=False)):
            raw_file_id = start_raw_id + offset
            stem = str(row.raw_file_stem)
            scheme = str(row.reporter_scheme)
            record = {
                "raw_file_stem": stem,
                "source_cohort": "newly_available",
                "stage14_action": "process_novel_mzml",
                "batch_chromatography": row.batch_chromatography,
                "valid_target_cells": int(row.valid_target_cells),
                "reporter_scheme": scheme,
                "raw_path": row.canonical_raw_path,
                "mzml_path": row.canonical_mzml_path,
                "raw_file_id": raw_file_id,
            }
            logger.info(
                "Stage 14 | novel run %d/%d | %s (%s)",
                offset + 1,
                len(novel),
                stem,
                scheme,
            )
            try:
                header = inspect_mzml_header(row.canonical_mzml_path)
                checksum_algorithm = str(header.get("source_checksum_algorithm", "sha1"))
                raw_checksum = file_checksum(row.canonical_raw_path, checksum_algorithm)
                if raw_checksum != header.get("source_checksum_embedded"):
                    raise AssertionError("RAW checksum differs from mzML embedded source checksum")
                reporter_masses = {
                    str(channel): float(mass)
                    for channel, mass in config["reporter_schemes"][scheme]["masses"].items()
                }
                run_psms = psm_lookup[psm_lookup["raw_file_stem"].eq(stem)]
                embeddings, reporters, metadata, counts, reporter_qc = process_mzml_run(
                    row.canonical_mzml_path,
                    stem,
                    raw_file_id,
                    reporter_masses,
                    stage,
                    run_psms,
                )
                if counts["parsed_spectra"] != int(header["declared_spectrum_count"]):
                    raise AssertionError("parsed spectrum count differs from mzML header")
                if int(row.valid_target_cells) > 0 and metadata.empty:
                    raise AssertionError("target-containing run has no retained spectra")
                start, end = _append_dataset(embedding_ds, embeddings)
                _append_dataset(reporter_ds, reporters)
                precursor = metadata[[
                    "precursor_mz",
                    "precursor_charge",
                    "retention_time_minutes",
                    "fragment_total_intensity",
                ]].to_numpy(dtype=np.float32)
                precursor[:, 3] = np.log1p(precursor[:, 3])
                _append_dataset(precursor_ds, precursor)
                _append_dataset(scan_ds, metadata["scan_id"].to_numpy(dtype=np.int64))
                _append_dataset(raw_id_ds, np.full(len(metadata), raw_file_id, dtype=np.int32))
                _append_dataset(reporter_scan_ds, metadata["scan_id"].to_numpy(dtype=np.int64))
                _append_dataset(
                    reporter_raw_id_ds,
                    np.full(len(metadata), raw_file_id, dtype=np.int32),
                )
                metadata.insert(0, "spectrum_id", np.arange(start, end, dtype=np.int64))
                metadata_frames.append(metadata)
                new_qc_frames.append(reporter_qc)
                software = header.get("software") or [{}]
                methods = header.get("processing_methods") or []
                record.update({
                    "processing_status": "processed_new",
                    "status_reason": None,
                    "conversion_tool": software[0].get("id"),
                    "conversion_tool_version": software[0].get("version"),
                    "conversion_methods": "; ".join(
                        method
                        for item in methods
                        for method in item.get("methods", [])
                    ),
                    "raw_checksum_algorithm": checksum_algorithm,
                    "raw_checksum": raw_checksum,
                    "raw_checksum_matches_mzml": True,
                    "mzml_sha256": file_checksum(row.canonical_mzml_path, "sha256"),
                    "spectrum_id_start": start,
                    "spectrum_id_end": end,
                    "identified_psm_scans": int(metadata["has_psm"].sum()),
                    **counts,
                })
            except Exception as error:
                logger.exception("Stage 14 | failed novel run %s", stem)
                record.update({
                    "processing_status": "failed",
                    "status_reason": f"{type(error).__name__}: {error}",
                })
            new_records.append(record)
    new_log = pd.DataFrame(new_records)
    failed = new_log[
        new_log["processing_status"].eq("failed") & new_log["valid_target_cells"].gt(0)
    ]
    if not failed.empty:
        raise RuntimeError(
            "one or more novel target-containing runs failed: "
            f"{failed['raw_file_stem'].tolist()}"
        )
    spectra_tmp.replace(spectra_path)
    reporter_tmp.replace(reporter_path)
    metadata = pd.concat(metadata_frames, ignore_index=True, sort=False)
    metadata.to_parquet(metadata_path, index=False)

    old_log = pd.read_csv(stage["stage11_processing_log"])
    old_log["source_cohort"] = "existing_stage11_scope"
    old_log["stage14_action"] = np.where(
        old_log["processing_status"].eq("processed"),
        "reused_stage11_store",
        "preserved_stage11_exclusion",
    )
    processing_log = pd.concat([old_log, new_log], ignore_index=True, sort=False)
    old_qc = pd.read_csv(stage["stage11_reporter_qc"])
    reporter_qc = pd.concat([old_qc, *new_qc_frames], ignore_index=True, sort=False)
    return processing_log, metadata, reporter_qc, spectra_path, reporter_path


def build_spectrum_bags_with_audit(
    manifest: pd.DataFrame,
    spectrum_metadata: pd.DataFrame,
    reporter_store_path: str | Path,
    minimum_reporter_intensity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build nonempty target-cell spectrum bags and retain explicit failures."""
    candidates = manifest[manifest["stage13_candidate"]].copy()
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    metadata_by_run = {
        stem: frame for stem, frame in spectrum_metadata.groupby("raw_file_stem", sort=False)
    }
    with h5py.File(reporter_store_path, "r") as store:
        reporter = store["reporter_intensities"]
        for cell in candidates.itertuples(index=False):
            run = metadata_by_run.get(str(cell.raw_file_stem))
            if run is None or run.empty:
                exclusions.append({
                    "sample_id": cell.sample_id,
                    "modality": "spectrum",
                    "reason": "raw_file_has_no_retained_spectra",
                })
                continue
            spectrum_ids = run["spectrum_id"].to_numpy(dtype=np.int64)
            channel_index = int(str(cell.reporter_channel).removeprefix("RI")) - 1
            intensities = np.asarray(reporter[spectrum_ids, channel_index], dtype=np.float32)
            keep = intensities >= minimum_reporter_intensity
            selected_ids = spectrum_ids[keep]
            selected_intensities = intensities[keep]
            if selected_ids.size == 0:
                exclusions.append({
                    "sample_id": cell.sample_id,
                    "modality": "spectrum",
                    "reason": "no_positive_reporter_spectra",
                })
                continue
            records.append({
                "bag_id": len(records),
                "sample_id": cell.sample_id,
                "label_raw": cell.label_raw,
                "label_binary": int(cell.label_raw == "sc_u"),
                "raw_file_stem": cell.raw_file_stem,
                "reporter_channel": cell.reporter_channel,
                "batch_digest": cell.batch_digest,
                "batch_sort": cell.batch_sort,
                "batch_chromatography": cell.batch_chromatography,
                "spectrum_count": int(selected_ids.size),
                "total_reporter_intensity": float(selected_intensities.sum()),
                "median_reporter_intensity": float(np.median(selected_intensities)),
                "spectrum_ids": selected_ids.tolist(),
            })
    return pd.DataFrame(records), pd.DataFrame(
        exclusions, columns=["sample_id", "modality", "reason"]
    )


def attach_full_psm_metadata(
    spectrum_metadata: pd.DataFrame,
    full_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach stable modified-peptide/protein identifiers and QC flags to spectra."""
    extra = full_lookup[[
        "raw_file_stem",
        "scan_id",
        "psm_sequence_full",
        "psm_modified_sequence",
        "psm_leading_razor_protein",
        "psm_leading_proteins",
        "psm_reverse",
        "psm_potential_contaminant",
        "psm_q_value_full",
        "psm_pep_full",
        "psm_precursor_ion_fraction_full",
        "psm_score_full",
    ]]
    enriched = spectrum_metadata.merge(
        extra,
        on=["raw_file_stem", "scan_id"],
        how="left",
        validate="one_to_one",
    )
    enriched["peptide_id_stable"] = enriched["psm_modified_sequence"].fillna(
        enriched["psm_sequence_full"]
    )
    enriched["protein_id_stable"] = enriched["psm_leading_proteins"].fillna(
        enriched["psm_leading_razor_protein"]
    )
    enriched["has_full_psm"] = enriched["peptide_id_stable"].notna()
    return enriched


def aggregate_reporter_features(
    spectrum_metadata: pd.DataFrame,
    bags: pd.DataFrame,
    reporter_store_path: str | Path,
    maximum_pep: float,
    minimum_reporter_intensity: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate high-quality PSM reporter signals into peptide/protein matrices."""
    metadata = spectrum_metadata.set_index("spectrum_id", drop=False)
    pep = pd.to_numeric(metadata["psm_pep_full"], errors="coerce")
    quality = pep.notna() & pep.le(maximum_pep)
    quality &= ~is_flagged(metadata["psm_reverse"])
    quality &= ~is_flagged(metadata["psm_potential_contaminant"])
    quality &= metadata["peptide_id_stable"].notna() & metadata["protein_id_stable"].notna()
    quality_ids = set(metadata.index[quality].astype(int))
    evidence_frames: list[pd.DataFrame] = []
    with h5py.File(reporter_store_path, "r") as store:
        reporter = store["reporter_intensities"]
        for bag in bags.itertuples(index=False):
            spectrum_ids = np.asarray(bag.spectrum_ids, dtype=np.int64)
            selected = np.asarray([value in quality_ids for value in spectrum_ids], dtype=bool)
            spectrum_ids = spectrum_ids[selected]
            if spectrum_ids.size == 0:
                continue
            channel_index = int(str(bag.reporter_channel).removeprefix("RI")) - 1
            intensities = np.asarray(reporter[spectrum_ids, channel_index], dtype=np.float64)
            positive = np.isfinite(intensities) & (intensities >= minimum_reporter_intensity)
            spectrum_ids = spectrum_ids[positive]
            intensities = intensities[positive]
            if spectrum_ids.size == 0:
                continue
            selected_metadata = metadata.loc[spectrum_ids]
            frame = pd.DataFrame({
                "sample_id": bag.sample_id,
                "raw_file_stem": bag.raw_file_stem,
                "reporter_channel": bag.reporter_channel,
                "peptide": selected_metadata["peptide_id_stable"].astype(str).to_numpy(),
                "protein": selected_metadata["protein_id_stable"].astype(str).to_numpy(),
                "intensity": intensities,
            })
            aggregated = frame.groupby(
                ["sample_id", "raw_file_stem", "reporter_channel", "peptide", "protein"],
                observed=True,
                as_index=False,
            ).agg(
                intensity=("intensity", "sum"),
                psm_observations=("intensity", "size"),
            )
            evidence_frames.append(aggregated)
    if not evidence_frames:
        raise ValueError("no high-quality channel-specific PSM evidence was reconstructed")
    evidence = pd.concat(evidence_frames, ignore_index=True)
    peptide = evidence.groupby(["peptide", "sample_id"], observed=True)["intensity"].sum().unstack("sample_id")
    protein = evidence.groupby(["protein", "sample_id"], observed=True)["intensity"].sum().unstack("sample_id")
    peptide = peptide.sort_index().astype(np.float32)
    protein = protein.sort_index().astype(np.float32)
    qc = evidence.groupby("sample_id", observed=True).agg(
        detected_peptides=("peptide", "nunique"),
        detected_proteins=("protein", "nunique"),
        quality_psm_observations=("psm_observations", "sum"),
        total_peptide_reporter_intensity=("intensity", "sum"),
    ).reset_index()
    return evidence, peptide, protein, qc


def match_modalities(
    manifest: pd.DataFrame,
    spectrum_bags: pd.DataFrame,
    peptide_matrix: pd.DataFrame,
    protein_matrix: pd.DataFrame,
    feature_qc: pd.DataFrame,
    spectrum_exclusions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Intersect candidate cells across spectrum, peptide, and protein modalities."""
    candidates = manifest[manifest["stage13_candidate"]].copy()
    bag_lookup = spectrum_bags.set_index("sample_id")
    qc_lookup = feature_qc.set_index("sample_id")
    spectrum_reason = (
        spectrum_exclusions.set_index("sample_id")["reason"].to_dict()
        if not spectrum_exclusions.empty else {}
    )
    records = []
    for cell in candidates.itertuples(index=False):
        sample_id = str(cell.sample_id)
        has_spectrum = sample_id in bag_lookup.index
        has_peptide = sample_id in peptide_matrix.columns and peptide_matrix[sample_id].notna().any()
        has_protein = sample_id in protein_matrix.columns and protein_matrix[sample_id].notna().any()
        reasons = []
        if not has_spectrum:
            reasons.append(spectrum_reason.get(sample_id, "missing_spectrum_bag"))
        if not has_peptide:
            reasons.append("missing_peptide_vector")
        if not has_protein:
            reasons.append("missing_protein_vector")
        qc_row = qc_lookup.loc[sample_id] if sample_id in qc_lookup.index else None
        records.append({
            "sample_id": sample_id,
            "label_raw": cell.label_raw,
            "raw_file_stem": cell.raw_file_stem,
            "reporter_channel": cell.reporter_channel,
            "batch_chromatography": cell.batch_chromatography,
            "source_cohort": cell.source_cohort,
            "has_spectrum_bag": has_spectrum,
            "has_peptide_vector": has_peptide,
            "has_protein_vector": has_protein,
            "spectrum_count": int(bag_lookup.loc[sample_id, "spectrum_count"]) if has_spectrum else 0,
            "detected_peptides": int(qc_row["detected_peptides"]) if qc_row is not None else 0,
            "detected_proteins": int(qc_row["detected_proteins"]) if qc_row is not None else 0,
            "quality_psm_observations": int(qc_row["quality_psm_observations"]) if qc_row is not None else 0,
            "total_peptide_reporter_intensity": float(qc_row["total_peptide_reporter_intensity"]) if qc_row is not None else np.nan,
            "match_status": "matched" if not reasons else "excluded",
            "exclusion_reason": ";".join(reasons) if reasons else None,
        })
    audit = pd.DataFrame(records)
    matched_ids = audit.loc[audit["match_status"].eq("matched"), "sample_id"].tolist()
    matched_manifest = candidates[candidates["sample_id"].isin(matched_ids)].copy()
    matched_manifest = matched_manifest.merge(
        audit[[
            "sample_id", "spectrum_count", "detected_peptides", "detected_proteins",
            "quality_psm_observations", "total_peptide_reporter_intensity",
        ]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    matched_manifest["is_valid_target"] = True
    matched_bags = spectrum_bags[spectrum_bags["sample_id"].isin(matched_ids)].copy()
    matched_bags = matched_bags.sort_values("sample_id").reset_index(drop=True)
    matched_bags["bag_id"] = np.arange(len(matched_bags), dtype=np.int64)
    matched_peptide = peptide_matrix.reindex(columns=matched_ids)
    matched_protein = protein_matrix.reindex(columns=matched_ids)
    return audit, matched_manifest, matched_bags, matched_peptide, matched_protein


def build_stage14_splits(
    matched_manifest: pd.DataFrame,
    evaluation_config: dict[str, Any],
    split_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build immutable grouped outer, inner, and batch-holdout assignments."""
    samples = matched_manifest[[
        "sample_id", "label_raw", "raw_file_stem", "batch_chromatography"
    ]].copy()
    group_column = "raw_file_stem"
    outer = make_outer_assignments(
        samples,
        group_column,
        int(evaluation_config["outer_folds"]),
        int(evaluation_config["seed"]),
    )
    inner = make_inner_assignments(
        outer,
        group_column,
        int(evaluation_config["inner_folds"]),
        int(evaluation_config["seed"]),
    )
    batch = make_batch_holdouts(samples, str(evaluation_config["batch_holdout_column"]))
    split_directory.mkdir(parents=True, exist_ok=True)
    outer.to_csv(split_directory / "stage_14_outer_fold_assignments.csv", index=False)
    inner.to_csv(split_directory / "stage_14_inner_fold_assignments.csv", index=False)
    batch.to_csv(split_directory / "stage_14_batch_holdout_assignments.csv", index=False)
    return outer, inner, batch


def validate_stage14(
    spectra_path: Path,
    reporter_path: Path,
    spectrum_metadata: pd.DataFrame,
    bags: pd.DataFrame,
    peptide_matrix: pd.DataFrame,
    protein_matrix: pd.DataFrame,
    matched_manifest: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    number_of_bins: int,
) -> None:
    """Enforce all Stage 14 multimodal alignment and leakage gates."""
    validate_spectrum_outputs(
        spectra_path,
        reporter_path,
        spectrum_metadata,
        bags,
        number_of_bins,
    )
    manifest_ids = matched_manifest["sample_id"].astype(str).tolist()
    if len(manifest_ids) != len(set(manifest_ids)):
        raise AssertionError("matched manifest contains duplicate sample IDs")
    if set(manifest_ids) != set(bags["sample_id"].astype(str)):
        raise AssertionError("matched manifest and spectrum bags contain different cells")
    if set(manifest_ids) != set(peptide_matrix.columns.astype(str)):
        raise AssertionError("matched manifest and peptide matrix contain different cells")
    if set(manifest_ids) != set(protein_matrix.columns.astype(str)):
        raise AssertionError("matched manifest and protein matrix contain different cells")
    if peptide_matrix.columns.has_duplicates or protein_matrix.columns.has_duplicates:
        raise AssertionError("feature matrix contains duplicate sample columns")
    if peptide_matrix.index.has_duplicates or protein_matrix.index.has_duplicates:
        raise AssertionError("feature matrix contains duplicate feature identifiers")
    if peptide_matrix.notna().sum(axis=0).eq(0).any():
        raise AssertionError("a matched cell has zero detected peptides")
    if protein_matrix.notna().sum(axis=0).eq(0).any():
        raise AssertionError("a matched cell has zero detected proteins")
    for _, frame in outer.groupby("outer_fold"):
        assert_group_disjoint(frame, "raw_file_stem")
    for _, frame in inner.groupby(["outer_fold", "inner_fold"]):
        assert_group_disjoint(frame, "raw_file_stem")
    if set(outer["sample_id"]) != set(manifest_ids):
        raise AssertionError("outer splits do not cover the complete matched corpus")


def render_report(summary: dict[str, Any]) -> str:
    """Render the Stage 14 matching and QC report."""
    return "\n".join([
        "# Stage 14 — Matched protein vectors and spectrum bags",
        "",
        "## Matched multimodal corpus",
        "",
        f"- Stage 13 candidate cells: {summary['stage13_candidate_cells']:,}.",
        f"- Matched cells: **{summary['matched_cells']:,}** (`sc_m0` = {summary['label_counts'].get('sc_m0', 0):,}; `sc_u` = {summary['label_counts'].get('sc_u', 0):,}).",
        f"- Excluded during Stage 14 matching: {summary['stage14_excluded_cells']:,}.",
        f"- Matched raw-file groups: {summary['matched_raw_file_groups']:,}.",
        f"- Chromatography batches: {', '.join(summary['batches'])}.",
        "",
        "## Spectrum reconstruction",
        "",
        f"- Existing Stage 11 runs reused: {summary['reused_runs']:,}.",
        f"- Novel mzML runs processed: {summary['novel_runs_processed']:,}.",
        f"- Failed target-containing runs: {summary['failed_runs']:,}.",
        f"- Shared retained MS2 spectra: {summary['retained_spectra']:,}.",
        f"- Spectra with a linked search-result PSM: {summary['identified_psm_spectra']:,}.",
        f"- Spectrum dimensions: {summary['spectrum_dimensions']:,}.",
        f"- Spectrum bag sizes: minimum {summary['bag_size_minimum']:,}, median {summary['bag_size_median']:,.0f}, maximum {summary['bag_size_maximum']:,}.",
        "- Each shared fragment spectrum is stored once. Reporter intensities remain in a separate aligned 16-channel store.",
        "",
        "## Peptide and protein reconstruction",
        "",
        f"- Stable peptide features: {summary['peptide_features']:,}.",
        f"- Stable protein features: {summary['protein_features']:,}.",
        f"- Aggregated peptide/protein/cell evidence rows: {summary['aggregated_evidence_rows']:,}.",
        f"- PSM QC: PEP ≤ {summary['maximum_pep']}; reverse and potential-contaminant identifications excluded.",
        "- Reporter intensities were extracted from mzML spectra for RI1–RI16, so extended TMTpro channels do not depend on missing corrected-intensity columns in the combined search table.",
        "- Missing matrix values represent non-detection. Prevalence filtering, imputation, and scaling remain training-fold operations.",
        "",
        "## Leakage-safe splits",
        "",
        f"- Outer folds: {summary['outer_folds']}; inner folds per outer training set: {summary['inner_folds']}.",
        "- Every raw-file group remains wholly inside one train, validation, or test partition within each fold.",
        "- Every outer test and inner validation partition contains both biological labels.",
        "",
        "## Stage gate",
        "",
        "Stage 14 passes when every matched cell has one valid label, one nonempty protein vector, one nonempty peptide vector, and one nonempty spectrum bag; all HDF5, Parquet, manifest, and split references align exactly; and the complete project test suite passes.",
        "",
    ])


def run_stage_fourteen(
    config_path: str | Path = "configs/spectrum_mil.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute Stage 14 and write the complete matched multimodal corpus."""
    config = load_yaml_config(config_path)
    stage = config["stage_14"]
    preprocessing = load_yaml_config(stage["preprocessing_config"])
    evaluation = load_yaml_config(stage["evaluation_config"])
    manifest = pd.read_parquet(stage["expanded_manifest_path"])
    run_summary = pd.read_csv(stage["run_summary_path"])
    candidate_count = int(manifest["stage13_candidate"].sum())
    full_lookup = load_full_psm_lookup(
        stage["evidence_path"],
        set(run_summary["raw_file_stem"].astype(str)),
    )
    processing_log, spectrum_metadata, reporter_qc, spectra_path, reporter_path = append_novel_spectra(
        config,
        stage,
        manifest,
        run_summary,
        full_lookup,
        dry_run=dry_run,
    )
    if dry_run:
        return {
            "dry_run": True,
            "stage13_candidate_cells": candidate_count,
            "novel_runs": int(run_summary["acquisition_status"].eq("ready_new_stage13").sum()),
            "search_result_scans": int(len(full_lookup)),
        }
    spectrum_bags, spectrum_exclusions = build_spectrum_bags_with_audit(
        manifest,
        spectrum_metadata,
        reporter_path,
        float(stage["filtering"]["minimum_cell_reporter_intensity"]),
    )
    spectrum_metadata = attach_full_psm_metadata(spectrum_metadata, full_lookup)
    evidence, peptide, protein, feature_qc = aggregate_reporter_features(
        spectrum_metadata,
        spectrum_bags,
        reporter_path,
        float(preprocessing["quality_control"]["max_pep"]),
        float(stage["filtering"]["minimum_cell_reporter_intensity"]),
    )
    audit, matched_manifest, matched_bags, matched_peptide, matched_protein = match_modalities(
        manifest,
        spectrum_bags,
        peptide,
        protein,
        feature_qc,
        spectrum_exclusions,
    )
    if matched_manifest.empty:
        raise ValueError("Stage 14 produced no matched multimodal cells")
    output_directory = Path(stage["output_directory"])
    split_directory = Path(stage["split_directory"])
    table_directory = Path("outputs/tables")
    report_directory = Path("outputs/reports")
    for directory in [output_directory, split_directory, table_directory, report_directory]:
        directory.mkdir(parents=True, exist_ok=True)
    outer, inner, batch = build_stage14_splits(matched_manifest, evaluation, split_directory)
    number_of_bins = int(np.ceil(
        (float(stage["representation"]["maximum_mz"]) - float(stage["representation"]["minimum_mz"]))
        / float(stage["representation"]["bin_width_mz"])
    ))
    validate_stage14(
        spectra_path,
        reporter_path,
        spectrum_metadata,
        matched_bags,
        matched_peptide,
        matched_protein,
        matched_manifest,
        outer,
        inner,
        number_of_bins,
    )

    spectrum_metadata.to_parquet(output_directory / "spectrum_metadata.parquet", index=False)
    matched_bags.to_parquet(output_directory / "spectrum_bag_index.parquet", index=False)
    evidence.to_parquet(output_directory / "channel_peptide_protein_evidence.parquet", index=False)
    matched_peptide.to_parquet(output_directory / "peptide_by_sample.parquet")
    matched_protein.to_parquet(output_directory / "protein_by_sample.parquet")
    matched_manifest.to_parquet(output_directory / "matched_multimodal_manifest.parquet", index=False)

    processing_log.to_csv(table_directory / "stage_14_processing_log.csv", index=False)
    reporter_qc.to_csv(table_directory / "stage_14_reporter_qc.csv", index=False)
    audit.to_csv(table_directory / "stage_14_cell_qc.csv", index=False)
    audit[audit["match_status"].eq("excluded")].to_csv(
        table_directory / "stage_14_multimodal_matching_audit.csv", index=False
    )
    class_batch = matched_manifest.groupby(
        ["source_cohort", "batch_chromatography", "batch_sort", "batch_digest", "label_raw"],
        dropna=False,
    ).agg(
        cell_count=("sample_id", "size"),
        raw_file_groups=("raw_file_stem", "nunique"),
        peptide_count_median=("detected_peptides", "median"),
        protein_count_median=("detected_proteins", "median"),
        spectrum_count_median=("spectrum_count", "median"),
    ).reset_index()
    class_batch.to_csv(table_directory / "stage_14_class_batch_counts.csv", index=False)
    outer_summary = outer.groupby(
        ["outer_fold", "partition", "label_raw"], dropna=False
    ).size().rename("cell_count").reset_index()
    inner_summary = inner.groupby(
        ["outer_fold", "inner_fold", "partition", "label_raw"], dropna=False
    ).size().rename("cell_count").reset_index()
    batch_summary = batch.groupby(
        ["held_out_batch", "partition", "label_raw"], dropna=False
    ).size().rename("cell_count").reset_index()
    outer_summary.to_csv(table_directory / "stage_14_outer_fold_summary.csv", index=False)
    inner_summary.to_csv(table_directory / "stage_14_inner_fold_summary.csv", index=False)
    batch_summary.to_csv(table_directory / "stage_14_batch_holdout_summary.csv", index=False)

    label_counts = matched_manifest["label_raw"].value_counts().sort_index().to_dict()
    processed_new = processing_log["processing_status"].eq("processed_new")
    summary = {
        "stage13_candidate_cells": candidate_count,
        "matched_cells": int(len(matched_manifest)),
        "stage14_excluded_cells": int(candidate_count - len(matched_manifest)),
        "label_counts": label_counts,
        "matched_raw_file_groups": int(matched_manifest["raw_file_stem"].nunique()),
        "batches": sorted(matched_manifest["batch_chromatography"].astype(str).unique()),
        "reused_runs": int(processing_log["stage14_action"].eq("reused_stage11_store").sum()),
        "novel_runs_processed": int(processed_new.sum()),
        "failed_runs": int(processing_log["processing_status"].eq("failed").sum()),
        "retained_spectra": int(len(spectrum_metadata)),
        "identified_psm_spectra": int(spectrum_metadata["has_full_psm"].sum()),
        "spectrum_dimensions": number_of_bins,
        "bag_size_minimum": int(matched_bags["spectrum_count"].min()),
        "bag_size_median": float(matched_bags["spectrum_count"].median()),
        "bag_size_maximum": int(matched_bags["spectrum_count"].max()),
        "peptide_features": int(len(matched_peptide)),
        "protein_features": int(len(matched_protein)),
        "aggregated_evidence_rows": int(len(evidence)),
        "maximum_pep": float(preprocessing["quality_control"]["max_pep"]),
        "outer_folds": int(evaluation["outer_folds"]),
        "inner_folds": int(evaluation["inner_folds"]),
        "outer_assignment_rows": int(len(outer)),
        "inner_assignment_rows": int(len(inner)),
        "spectra_h5_sha256": sha256sum(spectra_path),
        "reporter_h5_sha256": sha256sum(reporter_path),
        "matched_manifest_path": str(output_directory / "matched_multimodal_manifest.parquet"),
        "protein_matrix_path": str(output_directory / "protein_by_sample.parquet"),
        "peptide_matrix_path": str(output_directory / "peptide_by_sample.parquet"),
        "spectrum_bag_index_path": str(output_directory / "spectrum_bag_index.parquet"),
    }
    (table_directory / "stage_14_run_metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (report_directory / "stage_14_matched_multimodal_corpus.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spectrum_mil.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_14_matched_multimodal_corpus.log"))
    summary = run_stage_fourteen(args.config, dry_run=args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 14 completed: %s", summary)


if __name__ == "__main__":
    main()
