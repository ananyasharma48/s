"""Run Stage 11 full raw-spectrum preprocessing and build cell spectrum bags."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymzml

from src.data.extract_reporter_ions import (
    extract_reporters_from_peaks,
    file_checksum,
    inspect_mzml_header,
    validate_reporter_scheme,
)
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


MAX_REPORTER_CHANNELS = 16
PSM_COLUMNS = [
    "Raw.file",
    "MS.MS.scan.number",
    "Sequence",
    "Leading.razor.protein",
    "dart_qval",
    "PEP",
    "PIF",
    "Score",
]


def discover_unique_files(roots: list[str | Path], suffix: str) -> dict[str, Path]:
    """Find files recursively and reject ambiguous duplicate stems."""
    discovered: dict[str, Path] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            raise FileNotFoundError(f"configured data root does not exist: {root}")
        for path in root.rglob(f"*{suffix}"):
            if path.stem in discovered and path.resolve() != discovered[path.stem].resolve():
                raise ValueError(f"duplicate {suffix} stem found: {path.stem}")
            discovered[path.stem] = path
    return discovered


def build_run_inventory(
    manifest: pd.DataFrame,
    raw_files: dict[str, Path],
    mzml_files: dict[str, Path],
    skipped: dict[str, str],
    scheme_by_batch: dict[str, str],
) -> pd.DataFrame:
    """Create one auditable processing row for every manifest raw-file group."""
    required = {"raw_file_stem", "batch_chromatography", "is_valid_target"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing run fields: {sorted(missing)}")
    grouped = manifest.groupby("raw_file_stem", sort=True).agg(
        batch_chromatography=("batch_chromatography", "first"),
        manifest_channels=("reporter_channel", "nunique"),
        valid_target_cells=("is_valid_target", "sum"),
    ).reset_index()
    records = []
    for row in grouped.itertuples(index=False):
        stem = str(row.raw_file_stem)
        batch = str(row.batch_chromatography)
        if batch not in scheme_by_batch:
            raise ValueError(f"no reporter scheme configured for batch {batch}")
        raw_path = raw_files.get(stem)
        mzml_path = mzml_files.get(stem)
        if stem in skipped:
            status, reason = "skipped_user", skipped[stem]
        elif raw_path is None:
            status, reason = "missing_raw", "RAW file not found"
        elif mzml_path is None:
            status, reason = "missing_mzml", "mzML conversion not found"
        else:
            status, reason = "ready", None
        records.append({
            "raw_file_stem": stem,
            "batch_chromatography": batch,
            "valid_target_cells": int(row.valid_target_cells),
            "manifest_channels": int(row.manifest_channels),
            "reporter_scheme": scheme_by_batch[batch],
            "raw_path": str(raw_path) if raw_path else None,
            "mzml_path": str(mzml_path) if mzml_path else None,
            "processing_status": status,
            "status_reason": reason,
        })
    return pd.DataFrame(records)


def bin_fragment_spectrum(
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    minimum_mz: float,
    maximum_mz: float,
    bin_width_mz: float,
    excluded_region: tuple[float, float] | None,
    intensity_transform: str,
    normalization: str,
) -> np.ndarray:
    """Convert centroid peaks into one normalized fixed-width shared vector."""
    if maximum_mz <= minimum_mz or bin_width_mz <= 0:
        raise ValueError("invalid spectrum binning range or width")
    number_of_bins = int(np.ceil((maximum_mz - minimum_mz) / bin_width_mz))
    mz_values = np.asarray(mz_values, dtype=float)
    intensity_values = np.asarray(intensity_values, dtype=float)
    mask = (
        np.isfinite(mz_values)
        & np.isfinite(intensity_values)
        & (intensity_values > 0)
        & (mz_values >= minimum_mz)
        & (mz_values < maximum_mz)
    )
    if excluded_region is not None:
        lower, upper = excluded_region
        mask &= ~((mz_values >= lower) & (mz_values <= upper))
    selected_mz = mz_values[mask]
    selected_intensity = intensity_values[mask]
    if intensity_transform == "square_root":
        selected_intensity = np.sqrt(selected_intensity)
    elif intensity_transform != "none":
        raise ValueError(f"unsupported intensity transform: {intensity_transform}")
    indices = np.floor((selected_mz - minimum_mz) / bin_width_mz).astype(int)
    vector = np.bincount(
        indices, weights=selected_intensity, minlength=number_of_bins
    )[:number_of_bins].astype(np.float32)
    if normalization == "l2":
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
    elif normalization == "max":
        maximum = float(vector.max())
        if maximum > 0:
            vector /= maximum
    elif normalization != "none":
        raise ValueError(f"unsupported spectrum normalization: {normalization}")
    return vector


def load_psm_lookup(evidence_path: str | Path, raw_file_stems: set[str]) -> pd.DataFrame:
    """Load one best identification record per raw-file/scan pair."""
    evidence = pd.read_csv(evidence_path, usecols=PSM_COLUMNS, low_memory=False)
    evidence = evidence[evidence["Raw.file"].isin(raw_file_stems)].copy()
    evidence["MS.MS.scan.number"] = pd.to_numeric(
        evidence["MS.MS.scan.number"], errors="coerce"
    )
    evidence = evidence.dropna(subset=["MS.MS.scan.number"])
    evidence["MS.MS.scan.number"] = evidence["MS.MS.scan.number"].astype(np.int64)
    for column in ["dart_qval", "PEP", "PIF", "Score"]:
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    evidence = evidence.sort_values(
        ["Raw.file", "MS.MS.scan.number", "dart_qval", "PEP", "Score"],
        ascending=[True, True, True, True, False],
        na_position="last",
    ).drop_duplicates(["Raw.file", "MS.MS.scan.number"], keep="first")
    return evidence.rename(columns={
        "Raw.file": "raw_file_stem",
        "MS.MS.scan.number": "scan_id",
        "Sequence": "psm_sequence",
        "Leading.razor.protein": "psm_leading_protein",
        "dart_qval": "psm_q_value",
        "PEP": "psm_pep",
        "PIF": "psm_precursor_ion_fraction",
        "Score": "psm_score",
    })


def process_mzml_run(
    mzml_path: str | Path,
    raw_file_stem: str,
    raw_file_id: int,
    reporter_masses: dict[str, float],
    config: dict[str, Any],
    run_psms: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Parse, filter, encode, and summarize one mzML file."""
    filtering = config["filtering"]
    representation = config["representation"]
    tolerance = float(config["reporter_tolerance_da"])
    minimum_peaks = int(filtering["minimum_fragment_peaks"])
    minimum_tic = float(filtering["minimum_fragment_total_intensity"])
    minimum_reporter_sum = float(filtering["minimum_reporter_sum"])
    psm_q_max = filtering.get("maximum_psm_q_value")
    psm_q_max = None if psm_q_max is None else float(psm_q_max)
    excluded = representation.get("excluded_reporter_region")
    excluded_region = tuple(map(float, excluded)) if excluded is not None else None
    number_of_bins = int(np.ceil(
        (float(representation["maximum_mz"]) - float(representation["minimum_mz"]))
        / float(representation["bin_width_mz"])
    ))
    counts = {
        "parsed_spectra": 0,
        "ms1_scans": 0,
        "ms2_scans": 0,
        "other_ms_level_scans": 0,
        "filtered_invalid_precursor": 0,
        "filtered_too_few_fragment_peaks": 0,
        "filtered_low_fragment_intensity": 0,
        "filtered_low_reporter_sum": 0,
        "filtered_psm_q_value": 0,
    }
    embeddings: list[np.ndarray] = []
    reporter_vectors: list[np.ndarray] = []
    metadata_records: list[dict[str, Any]] = []
    reporter_match_counts = np.zeros(len(reporter_masses), dtype=np.int64)
    reporter_intensity_sums = np.zeros(len(reporter_masses), dtype=np.float64)
    reporter_errors: list[list[float]] = [[] for _ in reporter_masses]
    reader = pymzml.run.Reader(str(mzml_path), build_index_from_scratch=False)
    for spectrum in reader:
        counts["parsed_spectra"] += 1
        level = int(spectrum.ms_level)
        if level == 1:
            counts["ms1_scans"] += 1
            continue
        if level != 2:
            counts["other_ms_level_scans"] += 1
            continue
        counts["ms2_scans"] += 1
        scan_id = int(spectrum.ID)
        precursors = spectrum.selected_precursors
        precursor = precursors[0] if precursors else {}
        precursor_mz = float(precursor.get("mz", np.nan))
        precursor_charge = float(precursor.get("charge", np.nan))
        precursor_intensity = float(precursor.get("i", np.nan))
        if not np.isfinite(precursor_mz) or precursor_mz <= 0:
            counts["filtered_invalid_precursor"] += 1
            continue
        peaks = np.asarray(spectrum.peaks("centroided"), dtype=float)
        if peaks.ndim != 2 or peaks.shape[1] != 2 or len(peaks) < minimum_peaks:
            counts["filtered_too_few_fragment_peaks"] += 1
            continue
        fragment_tic = float(peaks[:, 1].sum())
        if not np.isfinite(fragment_tic) or fragment_tic < minimum_tic:
            counts["filtered_low_fragment_intensity"] += 1
            continue
        extracted = extract_reporters_from_peaks(
            peaks[:, 0], peaks[:, 1], reporter_masses, tolerance
        )
        reporter = np.zeros(MAX_REPORTER_CHANNELS, dtype=np.float32)
        for index, row in enumerate(extracted):
            reporter[index] = float(row["reporter_intensity"])
            if reporter[index] > 0:
                reporter_match_counts[index] += 1
                reporter_intensity_sums[index] += float(reporter[index])
                reporter_errors[index].append(float(row["mass_error_ppm"]))
        reporter_sum = float(reporter.sum())
        if reporter_sum < minimum_reporter_sum:
            counts["filtered_low_reporter_sum"] += 1
            continue
        embeddings.append(bin_fragment_spectrum(
            peaks[:, 0],
            peaks[:, 1],
            float(representation["minimum_mz"]),
            float(representation["maximum_mz"]),
            float(representation["bin_width_mz"]),
            excluded_region,
            str(representation["intensity_transform"]),
            str(representation["normalization"]),
        ))
        reporter_vectors.append(reporter)
        metadata_records.append({
            "raw_file_id": raw_file_id,
            "raw_file_stem": raw_file_stem,
            "scan_id": scan_id,
            "retention_time_minutes": float(spectrum.scan_time_in_minutes()),
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge,
            "precursor_intensity": precursor_intensity,
            "fragment_peak_count": int(len(peaks)),
            "fragment_total_intensity": fragment_tic,
            "reporter_intensity_sum": reporter_sum,
            "matched_reporter_count": int((reporter > 0).sum()),
        })
    if embeddings:
        embedding_array = np.vstack(embeddings).astype(representation["dtype"])
        reporter_array = np.vstack(reporter_vectors).astype(np.float32)
    else:
        embedding_array = np.empty((0, number_of_bins), dtype=representation["dtype"])
        reporter_array = np.empty((0, MAX_REPORTER_CHANNELS), dtype=np.float32)
    metadata = pd.DataFrame(metadata_records)
    if not metadata.empty:
        metadata = metadata.merge(
            run_psms,
            on=["raw_file_stem", "scan_id"],
            how="left",
            validate="one_to_one",
        )
        metadata["has_psm"] = metadata["psm_sequence"].notna()
        if psm_q_max is not None:
            keep = metadata["psm_q_value"].le(psm_q_max).fillna(False).to_numpy()
            counts["filtered_psm_q_value"] = int((~keep).sum())
            metadata = metadata.loc[keep].reset_index(drop=True)
            embedding_array = embedding_array[keep]
            reporter_array = reporter_array[keep]
    counts["retained_ms2_scans"] = int(len(metadata))
    qc_records = []
    for index, (channel, expected_mz) in enumerate(reporter_masses.items()):
        errors = np.asarray(reporter_errors[index], dtype=float)
        qc_records.append({
            "raw_file_stem": raw_file_stem,
            "reporter_channel": channel,
            "reporter_mz_expected": expected_mz,
            "retained_ms2_scans": int(len(metadata)),
            "matched_scans": int(reporter_match_counts[index]),
            "matched_scan_fraction": (
                float(reporter_match_counts[index] / len(metadata)) if len(metadata) else np.nan
            ),
            "total_reporter_intensity": float(reporter_intensity_sums[index]),
            "median_mass_error_ppm": float(np.median(errors)) if len(errors) else np.nan,
            "median_absolute_mass_error_ppm": float(np.median(np.abs(errors))) if len(errors) else np.nan,
        })
    return embedding_array, reporter_array, metadata, counts, pd.DataFrame(qc_records)


def _append_dataset(dataset: h5py.Dataset, values: np.ndarray) -> tuple[int, int]:
    start = int(dataset.shape[0])
    end = start + int(len(values))
    dataset.resize(end, axis=0)
    if len(values):
        dataset[start:end] = values
    return start, end


def build_cell_bags(
    manifest: pd.DataFrame,
    spectrum_metadata: pd.DataFrame,
    reporter_store_path: str | Path,
    minimum_reporter_intensity: float,
) -> pd.DataFrame:
    """Build one variable-sized spectrum ID collection per retained valid cell."""
    processed_runs = set(spectrum_metadata["raw_file_stem"])
    cells = manifest[
        manifest["is_valid_target"] & manifest["raw_file_stem"].isin(processed_runs)
    ].copy()
    if cells.empty:
        raise ValueError("no valid cells map to processed spectrum runs")
    records = []
    with h5py.File(reporter_store_path, "r") as reporter_store:
        reporter_dataset = reporter_store["reporter_intensities"]
        for bag_id, cell in enumerate(cells.itertuples(index=False)):
            channel_index = int(str(cell.reporter_channel).removeprefix("RI")) - 1
            raw_spectra = spectrum_metadata[
                spectrum_metadata["raw_file_stem"].eq(cell.raw_file_stem)
            ]
            spectrum_ids = raw_spectra["spectrum_id"].to_numpy(dtype=np.int64)
            if len(spectrum_ids) == 0:
                raise AssertionError(f"cell's raw file has no retained spectra: {cell.sample_id}")
            intensities = reporter_dataset[spectrum_ids, channel_index]
            keep = np.asarray(intensities) >= minimum_reporter_intensity
            selected_ids = spectrum_ids[keep]
            selected_intensities = np.asarray(intensities)[keep]
            if len(selected_ids) == 0:
                raise AssertionError(f"cell has an empty spectrum bag: {cell.sample_id}")
            records.append({
                "bag_id": bag_id,
                "sample_id": cell.sample_id,
                "label_raw": cell.label_raw,
                "label_binary": int(str(cell.label_raw) == "sc_u"),
                "raw_file_stem": cell.raw_file_stem,
                "reporter_channel": cell.reporter_channel,
                "batch_digest": cell.batch_digest,
                "batch_sort": cell.batch_sort,
                "batch_chromatography": cell.batch_chromatography,
                "spectrum_count": int(len(selected_ids)),
                "total_reporter_intensity": float(selected_intensities.sum()),
                "median_reporter_intensity": float(np.median(selected_intensities)),
                "spectrum_ids": selected_ids.tolist(),
            })
    return pd.DataFrame(records)


def validate_spectrum_outputs(
    spectra_path: str | Path,
    reporter_path: str | Path,
    metadata: pd.DataFrame,
    bags: pd.DataFrame,
    number_of_bins: int,
) -> None:
    """Verify alignment, uniqueness, normalized embeddings, and bag references."""
    if metadata.duplicated(["raw_file_stem", "scan_id"]).any():
        raise AssertionError("duplicate raw-file/scan pairs in spectrum metadata")
    if not np.array_equal(metadata["spectrum_id"], np.arange(len(metadata))):
        raise AssertionError("spectrum IDs are not contiguous and row-aligned")
    with h5py.File(spectra_path, "r") as spectra, h5py.File(reporter_path, "r") as reporters:
        if spectra["spectrum_embedding"].shape != (len(metadata), number_of_bins):
            raise AssertionError("spectrum embedding store shape is inconsistent")
        if spectra["precursor_features"].shape != (len(metadata), 4):
            raise AssertionError("precursor feature store shape is inconsistent")
        if reporters["reporter_intensities"].shape != (len(metadata), MAX_REPORTER_CHANNELS):
            raise AssertionError("reporter intensity store shape is inconsistent")
        if not np.array_equal(spectra["scan_id"][:], metadata["scan_id"]):
            raise AssertionError("HDF5 and metadata scan IDs are misaligned")
        if not np.array_equal(reporters["scan_id"][:], metadata["scan_id"]):
            raise AssertionError("reporter and metadata scan IDs are misaligned")
        if not np.array_equal(
            reporters["raw_file_id"][:], spectra["raw_file_id"][:]
        ):
            raise AssertionError("spectrum and reporter raw-file IDs are misaligned")
        for start in range(0, len(metadata), 4096):
            embedding = spectra["spectrum_embedding"][start:start + 4096]
            reporter_values = reporters["reporter_intensities"][start:start + 4096]
            if not np.isfinite(embedding).all():
                raise AssertionError("spectrum embeddings contain non-finite values")
            norms = np.linalg.norm(embedding, axis=1)
            if (norms == 0).any() or not np.allclose(norms, 1.0, atol=2e-4):
                raise AssertionError("spectrum embeddings are empty or not L2 normalized")
            if not np.isfinite(reporter_values).all() or (reporter_values < 0).any():
                raise AssertionError("reporter intensity store contains invalid values")
        for ids in bags["spectrum_ids"]:
            array = np.asarray(ids, dtype=int)
            if len(array) == 0 or array.min() < 0 or array.max() >= len(metadata):
                raise AssertionError("cell bag contains an invalid spectrum reference")
    if bags["sample_id"].duplicated().any() or (bags["spectrum_count"] < 1).any():
        raise AssertionError("cell spectrum bags must be unique and nonempty")


def plot_processing_summary(
    run_log: pd.DataFrame,
    bags: pd.DataFrame,
    reporter_qc: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot retained spectra, bag sizes, and reporter recovery."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    processed = run_log[run_log["processing_status"].str.startswith("processed")]
    axes[0].bar(np.arange(len(processed)), processed["retained_ms2_scans"])
    axes[0].set_xlabel("Processed raw-file ID")
    axes[0].set_ylabel("Retained MS2 spectra")
    axes[0].set_title("Retained spectra per run")
    label_values = [
        bags.loc[bags["label_raw"].eq(label), "spectrum_count"]
        for label in sorted(bags["label_raw"].unique())
    ]
    axes[1].boxplot(label_values, tick_labels=sorted(bags["label_raw"].unique()))
    axes[1].set_ylabel("Spectra per cell bag")
    axes[1].set_title("Cell bag sizes")
    reporter_grouped = reporter_qc.groupby("reporter_channel", sort=False)[
        "matched_scan_fraction"
    ].median()
    axes[2].bar(reporter_grouped.index, reporter_grouped.values)
    axes[2].tick_params(axis="x", rotation=90)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Median matched-scan fraction")
    axes[2].set_title("Reporter recovery across runs")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_stage_eleven(
    config_path: str | Path = "configs/spectrum_mil.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute full preprocessing for every available, non-skipped mzML run."""
    config = load_yaml_config(config_path)
    full = config["full_preprocessing"]
    manifest_path = Path(full["manifest_path"])
    evidence_path = Path(full["evidence_path"])
    if not manifest_path.is_file() or not evidence_path.is_file():
        raise FileNotFoundError("Stage 11 manifest or evidence input is missing")
    manifest = pd.read_parquet(manifest_path)
    raw_files = discover_unique_files(full["raw_roots"], ".raw")
    mzml_files = discover_unique_files(full["mzml_roots"], ".mzML")
    skipped = {str(key): str(value) for key, value in full["skipped_raw_files"].items()}
    scheme_by_batch = {
        str(key): str(value) for key, value in full["reporter_scheme_by_batch"].items()
    }
    inventory = build_run_inventory(
        manifest, raw_files, mzml_files, skipped, scheme_by_batch
    )
    unexpected_unavailable = inventory[
        ~inventory["processing_status"].isin(["ready", "skipped_user"])
    ]
    if not unexpected_unavailable.empty:
        raise FileNotFoundError(
            "unapproved Stage 11 inputs are unavailable: "
            f"{unexpected_unavailable['raw_file_stem'].tolist()}"
        )
    ready = inventory[inventory["processing_status"].eq("ready")].copy()
    retained_cell_count = int(
        manifest[
            manifest["is_valid_target"]
            & manifest["raw_file_stem"].isin(ready["raw_file_stem"])
        ]["sample_id"].nunique()
    )
    for scheme in set(ready["reporter_scheme"]):
        masses = config["reporter_schemes"][scheme]["masses"]
        validate_reporter_scheme(
            {str(channel): float(mass) for channel, mass in masses.items()},
            float(full["reporter_tolerance_da"]),
        )
    if dry_run:
        return {
            "dry_run": True,
            "manifest_runs": int(len(inventory)),
            "ready_runs": int(len(ready)),
            "user_skipped_runs": int(inventory["processing_status"].eq("skipped_user").sum()),
            "retained_valid_cells": retained_cell_count,
            "reporter_schemes": ready["reporter_scheme"].value_counts().to_dict(),
        }
    logger = logging.getLogger("single_cell_proteomics_mil")
    logger.info("Stage 11 | loading compact PSM lookup")
    psm_lookup = load_psm_lookup(evidence_path, set(ready["raw_file_stem"]))
    output_directory = Path(full["output_directory"])
    table_directory = Path("outputs/tables")
    report_directory = Path("outputs/reports")
    figure_directory = Path("outputs/figures")
    for directory in (output_directory, table_directory, report_directory, figure_directory):
        directory.mkdir(parents=True, exist_ok=True)
    spectra_path = output_directory / "spectra.h5"
    reporter_path = output_directory / "reporter_intensities.h5"
    spectra_tmp = output_directory / "spectra.h5.tmp"
    reporter_tmp = output_directory / "reporter_intensities.h5.tmp"
    representation = full["representation"]
    number_of_bins = int(np.ceil(
        (float(representation["maximum_mz"]) - float(representation["minimum_mz"]))
        / float(representation["bin_width_mz"])
    ))
    compression = str(representation["compression"])
    metadata_frames: list[pd.DataFrame] = []
    reporter_qc_frames: list[pd.DataFrame] = []
    run_records = inventory[~inventory["processing_status"].eq("ready")].to_dict("records")
    raw_file_id_by_stem: dict[str, int] = {}
    with h5py.File(spectra_tmp, "w") as spectra_store, h5py.File(reporter_tmp, "w") as reporter_store:
        embedding_ds = spectra_store.create_dataset(
            "spectrum_embedding",
            shape=(0, number_of_bins),
            maxshape=(None, number_of_bins),
            chunks=(256, number_of_bins),
            dtype=representation["dtype"],
            compression=compression,
        )
        precursor_ds = spectra_store.create_dataset(
            "precursor_features", shape=(0, 4), maxshape=(None, 4),
            chunks=(2048, 4), dtype="float32", compression=compression,
        )
        scan_ds = spectra_store.create_dataset(
            "scan_id", shape=(0,), maxshape=(None,), chunks=(4096,), dtype="int64",
            compression=compression,
        )
        raw_id_ds = spectra_store.create_dataset(
            "raw_file_id", shape=(0,), maxshape=(None,), chunks=(4096,), dtype="int32",
            compression=compression,
        )
        reporter_ds = reporter_store.create_dataset(
            "reporter_intensities", shape=(0, MAX_REPORTER_CHANNELS),
            maxshape=(None, MAX_REPORTER_CHANNELS), chunks=(2048, MAX_REPORTER_CHANNELS),
            dtype="float32", compression=compression,
        )
        reporter_scan_ds = reporter_store.create_dataset(
            "scan_id", shape=(0,), maxshape=(None,), chunks=(4096,), dtype="int64",
            compression=compression,
        )
        reporter_raw_id_ds = reporter_store.create_dataset(
            "raw_file_id", shape=(0,), maxshape=(None,), chunks=(4096,), dtype="int32",
            compression=compression,
        )
        bin_centers = (
            float(representation["minimum_mz"])
            + (np.arange(number_of_bins) + 0.5) * float(representation["bin_width_mz"])
        )
        spectra_store.create_dataset("fragment_bin_centers_mz", data=bin_centers)
        reporter_store.create_dataset(
            "reporter_channels",
            data=np.asarray([f"RI{index}" for index in range(1, 17)], dtype="S4"),
        )
        for attribute, value in {
            "representation": json.dumps(representation),
            "filtering": json.dumps(full["filtering"]),
            "row_alignment": "spectrum_id equals HDF5 row index",
        }.items():
            spectra_store.attrs[attribute] = value
            reporter_store.attrs[attribute] = value
        for raw_file_id, row in enumerate(ready.sort_values("raw_file_stem").itertuples(index=False)):
            stem = str(row.raw_file_stem)
            raw_file_id_by_stem[stem] = raw_file_id
            scheme_name = str(row.reporter_scheme)
            reporter_masses = {
                str(channel): float(mass)
                for channel, mass in config["reporter_schemes"][scheme_name]["masses"].items()
            }
            logger.info(
                "Stage 11 | %s/%s processing %s (%s)",
                raw_file_id + 1, len(ready), stem, scheme_name,
            )
            run_record = row._asdict()
            try:
                header = inspect_mzml_header(row.mzml_path)
                embedded_algorithm = str(header.get("source_checksum_algorithm", "sha1"))
                raw_checksum = file_checksum(row.raw_path, embedded_algorithm)
                if raw_checksum != header.get("source_checksum_embedded"):
                    raise AssertionError("RAW checksum differs from mzML source checksum")
                run_psms = psm_lookup[psm_lookup["raw_file_stem"].eq(stem)]
                embeddings, reporters, metadata, counts, reporter_qc = process_mzml_run(
                    row.mzml_path,
                    stem,
                    raw_file_id,
                    reporter_masses,
                    full,
                    run_psms,
                )
                if counts["parsed_spectra"] != int(header["declared_spectrum_count"]):
                    raise AssertionError("parsed spectrum count differs from mzML header")
                if row.valid_target_cells > 0 and len(metadata) == 0:
                    raise AssertionError("target-containing run has no retained MS2 spectra")
                start, end = _append_dataset(embedding_ds, embeddings)
                _append_dataset(reporter_ds, reporters)
                precursor_features = metadata[[
                    "precursor_mz", "precursor_charge", "retention_time_minutes",
                    "fragment_total_intensity",
                ]].to_numpy(dtype=np.float32)
                precursor_features[:, 3] = np.log1p(precursor_features[:, 3])
                _append_dataset(precursor_ds, precursor_features)
                _append_dataset(scan_ds, metadata["scan_id"].to_numpy(dtype=np.int64))
                _append_dataset(raw_id_ds, np.full(len(metadata), raw_file_id, dtype=np.int32))
                _append_dataset(reporter_scan_ds, metadata["scan_id"].to_numpy(dtype=np.int64))
                _append_dataset(
                    reporter_raw_id_ds, np.full(len(metadata), raw_file_id, dtype=np.int32)
                )
                metadata.insert(0, "spectrum_id", np.arange(start, end, dtype=np.int64))
                metadata_frames.append(metadata)
                reporter_qc_frames.append(reporter_qc)
                run_record.update({
                    "raw_file_id": raw_file_id,
                    "processing_status": "processed" if len(metadata) else "processed_no_retained_spectra",
                    "status_reason": None,
                    "conversion_tool": header["software"][0]["id"],
                    "conversion_tool_version": header["software"][0]["version"],
                    "conversion_command": "not embedded in supplied mzML",
                    "conversion_methods": "; ".join(
                        method for item in header["processing_methods"] for method in item["methods"]
                    ),
                    "raw_checksum_algorithm": embedded_algorithm,
                    "raw_checksum": raw_checksum,
                    "raw_checksum_matches_mzml": True,
                    "mzml_sha256": file_checksum(row.mzml_path, "sha256"),
                    "spectrum_id_start": start,
                    "spectrum_id_end": end,
                    "identified_psm_scans": int(metadata["has_psm"].sum()) if len(metadata) else 0,
                    **counts,
                })
            except Exception as error:
                logger.exception("Stage 11 | failed %s", stem)
                run_record.update({
                    "raw_file_id": raw_file_id,
                    "processing_status": "failed",
                    "status_reason": f"{type(error).__name__}: {error}",
                })
            run_records.append(run_record)
    spectra_tmp.replace(spectra_path)
    reporter_tmp.replace(reporter_path)
    run_log = pd.DataFrame(run_records).sort_values("raw_file_stem").reset_index(drop=True)
    spectrum_metadata = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    reporter_qc = pd.concat(reporter_qc_frames, ignore_index=True) if reporter_qc_frames else pd.DataFrame()
    failed_target_runs = run_log[
        run_log["processing_status"].eq("failed") & run_log["valid_target_cells"].gt(0)
    ]
    if not failed_target_runs.empty:
        run_log.to_csv(table_directory / "stage_11_conversion_processing_log.csv", index=False)
        raise RuntimeError(
            "one or more target-containing runs failed: "
            f"{failed_target_runs['raw_file_stem'].tolist()}"
        )
    spectrum_metadata.to_parquet(output_directory / "spectrum_metadata.parquet", index=False)
    bags = build_cell_bags(
        manifest,
        spectrum_metadata,
        reporter_path,
        float(full["filtering"]["minimum_cell_reporter_intensity"]),
    )
    bags.to_parquet(output_directory / "cell_bag_index.parquet", index=False)
    validate_spectrum_outputs(
        spectra_path, reporter_path, spectrum_metadata, bags, number_of_bins
    )
    run_log.to_csv(table_directory / "stage_11_conversion_processing_log.csv", index=False)
    reporter_qc.to_csv(table_directory / "stage_11_reporter_qc.csv", index=False)
    bag_summary = bags.groupby(
        ["label_raw", "batch_chromatography"], dropna=False
    ).agg(
        cells=("sample_id", "count"),
        spectrum_count_minimum=("spectrum_count", "min"),
        spectrum_count_median=("spectrum_count", "median"),
        spectrum_count_mean=("spectrum_count", "mean"),
        spectrum_count_maximum=("spectrum_count", "max"),
    ).reset_index()
    bag_summary.to_csv(table_directory / "stage_11_cell_bag_summary.csv", index=False)
    plot_processing_summary(
        run_log, bags, reporter_qc, figure_directory / "stage_11_processing_summary.png"
    )
    processed = run_log[run_log["processing_status"].str.startswith("processed")]
    summary = {
        "manifest_runs": int(len(inventory)),
        "processed_runs": int(len(processed)),
        "user_skipped_runs": int(run_log["processing_status"].eq("skipped_user").sum()),
        "failed_runs": int(run_log["processing_status"].eq("failed").sum()),
        "retained_spectra": int(len(spectrum_metadata)),
        "embedding_dimensions": number_of_bins,
        "reporter_channels": MAX_REPORTER_CHANNELS,
        "retained_cells": int(len(bags)),
        "excluded_cells_in_skipped_runs": int(
            manifest[
                manifest["is_valid_target"]
                & manifest["raw_file_stem"].isin(skipped)
            ]["sample_id"].nunique()
        ),
        "minimum_bag_size": int(bags["spectrum_count"].min()),
        "median_bag_size": float(bags["spectrum_count"].median()),
        "maximum_bag_size": int(bags["spectrum_count"].max()),
        "identified_psm_spectra": int(spectrum_metadata["has_psm"].sum()),
        "spectra_h5": str(spectra_path),
        "reporter_intensities_h5": str(reporter_path),
        "spectrum_metadata": str(output_directory / "spectrum_metadata.parquet"),
        "cell_bag_index": str(output_directory / "cell_bag_index.parquet"),
    }
    (table_directory / "stage_11_run_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    label_counts = bags["label_raw"].value_counts().sort_index()
    scheme_counts = processed["reporter_scheme"].value_counts().sort_index()
    (report_directory / "stage_11_full_raw_preprocessing.md").write_text(
        "\n".join([
            "# Stage 11 — Full raw-spectrum preprocessing", "",
            "## Scope and status", "",
            f"- Processed mzML runs: {summary['processed_runs']}/{summary['manifest_runs']}.",
            f"- User-approved skips: 2 (`190222S_LCA9_X_FP94AA`, `190222S_LCA9_X_FP94AB`); these account for {summary['excluded_cells_in_skipped_runs']} processed-corpus cells.",
            f"- Failed runs: {summary['failed_runs']}.",
            f"- Reporter schemes: {', '.join(f'`{scheme}` = {count} runs' for scheme, count in scheme_counts.items())}.",
            "- Every processed mzML was linked to its local RAW by the source checksum embedded during conversion; converter version, processing methods, and mzML SHA-256 are recorded per run.", "",
            "## Spectrum processing", "",
            f"- Retained shared MS2 spectra: {summary['retained_spectra']:,}.",
            f"- Representation: square-root intensity, reporter region removed, {representation['bin_width_mz']} Da bins from {representation['minimum_mz']}–{representation['maximum_mz']} m/z, and L2 normalization.",
            f"- Fixed spectrum dimensions: {summary['embedding_dimensions']}.",
            "- Shared fragment embeddings are stored once per raw-file/scan; channel-specific reporter signals are stored separately in an aligned 16-column matrix.",
            f"- Spectra with identified PSMs: {summary['identified_psm_spectra']:,}; PSM filtering was not applied because `maximum_psm_q_value` is null.", "",
            "## Cell spectrum bags", "",
            f"- Retained valid target cells: {summary['retained_cells']} ({', '.join(f'`{label}` = {count}' for label, count in label_counts.items())}).",
            f"- Bag sizes: minimum {summary['minimum_bag_size']:,}, median {summary['median_bag_size']:,.0f}, maximum {summary['maximum_bag_size']:,} spectra.",
            "- A spectrum enters a cell bag only when that cell's reporter channel has positive signal at or above the configured threshold.",
            "- Carrier, reference, normalization, unused, and non-target channels are never emitted as target-cell bags.", "",
            "## Integrity checks", "",
            "- Spectrum IDs are unique and contiguous across HDF5 and Parquet outputs.",
            "- HDF5 scan/raw-file IDs align exactly with spectrum metadata rows.",
            "- Every retained cell has a nonempty bag and all spectrum references are valid.",
            "- Shared spectrum embeddings are not duplicated across reporter channels.", "",
            "## Stage gate", "",
            "Stage 11 passes for the explicitly scoped 33-run corpus when all project tests pass and the conversion/processing log contains no failed run.", "",
        ]),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spectrum_mil.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_11_full_raw_preprocessing.log"))
    result = run_stage_eleven(args.config, args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 11 completed: %s", result)


if __name__ == "__main__":
    main()
