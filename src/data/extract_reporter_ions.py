"""Execute the Stage 10 mzML parsing and TMT reporter-ion pilot."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymzml

from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


REPORTER_TABLE_COLUMNS = [
    "raw_file",
    "scan_id",
    "retention_time",
    "precursor_mz",
    "precursor_charge",
    "reporter_channel",
    "reporter_mz_expected",
    "reporter_mz_observed",
    "reporter_intensity",
    "mass_error_ppm",
]


def file_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    """Return a streaming checksum without modifying the source file."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def inspect_mzml_header(path: str | Path) -> dict[str, Any]:
    """Read conversion provenance and acquisition metadata before the spectra."""
    metadata: dict[str, Any] = {
        "software": [],
        "processing_methods": [],
        "instruments": [],
    }
    for event, element in ElementTree.iterparse(path, events=("start", "end")):
        tag = _local_name(element.tag)
        if event == "start" and tag == "mzML":
            metadata["mzml_id"] = element.attrib.get("id")
            metadata["mzml_version"] = element.attrib.get("version")
        elif event == "end" and tag == "sourceFile":
            metadata["source_file_name"] = element.attrib.get("name")
            metadata["source_file_location"] = element.attrib.get("location")
            for child in element:
                if _local_name(child.tag) != "cvParam":
                    continue
                if child.attrib.get("name") in {"SHA-1", "MD5"}:
                    metadata["source_checksum_algorithm"] = child.attrib["name"].lower().replace("-", "")
                    metadata["source_checksum_embedded"] = child.attrib.get("value")
        elif event == "end" and tag == "software":
            names = [
                child.attrib.get("name")
                for child in element
                if _local_name(child.tag) == "cvParam"
            ]
            metadata["software"].append({
                "id": element.attrib.get("id"),
                "version": element.attrib.get("version"),
                "names": [name for name in names if name],
            })
        elif event == "end" and tag == "processingMethod":
            methods = [
                child.attrib.get("name")
                for child in element
                if _local_name(child.tag) == "cvParam"
            ]
            metadata["processing_methods"].append({
                "order": element.attrib.get("order"),
                "software_ref": element.attrib.get("softwareRef"),
                "methods": [method for method in methods if method],
            })
        elif event == "end" and tag == "referenceableParamGroup":
            parameters = [
                {"name": child.attrib.get("name"), "value": child.attrib.get("value")}
                for child in element
                if _local_name(child.tag) == "cvParam"
            ]
            metadata["instruments"].extend(parameters)
        elif event == "start" and tag == "run":
            metadata["run_id"] = element.attrib.get("id")
            metadata["run_start_timestamp"] = element.attrib.get("startTimeStamp")
        elif event == "start" and tag == "spectrumList":
            metadata["declared_spectrum_count"] = int(element.attrib["count"])
            metadata["default_data_processing_ref"] = element.attrib.get("defaultDataProcessingRef")
            break
    return metadata


def validate_reporter_scheme(
    reporter_masses: dict[str, float],
    tolerance_da: float,
) -> pd.DataFrame:
    """Validate reporter labels, ordering, and non-overlapping extraction windows."""
    if not reporter_masses:
        raise ValueError("reporter scheme has no masses")
    if tolerance_da <= 0:
        raise ValueError("reporter tolerance must be positive")
    expected_channels = [f"RI{index}" for index in range(1, len(reporter_masses) + 1)]
    if list(reporter_masses) != expected_channels:
        raise ValueError(
            f"reporter channels must be consecutive and ordered: {expected_channels}"
        )
    masses = np.asarray(list(reporter_masses.values()), dtype=float)
    if not np.isfinite(masses).all() or not np.all(np.diff(masses) > 0):
        raise ValueError("reporter masses must be finite and strictly increasing")
    preceding_gap = np.concatenate(([np.nan], np.diff(masses)))
    following_gap = np.concatenate((np.diff(masses), [np.nan]))
    nonoverlap = np.isnan(preceding_gap) | (preceding_gap > 2 * tolerance_da)
    nonoverlap &= np.isnan(following_gap) | (following_gap > 2 * tolerance_da)
    if not nonoverlap.all():
        raise ValueError("reporter extraction windows overlap at the configured tolerance")
    return pd.DataFrame({
        "reporter_channel": expected_channels,
        "reporter_mz_expected": masses,
        "preceding_mass_gap_da": preceding_gap,
        "following_mass_gap_da": following_gap,
        "extraction_window_nonoverlap": nonoverlap,
    })


def extract_reporters_from_peaks(
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    reporter_masses: dict[str, float],
    tolerance_da: float,
) -> list[dict[str, float | str]]:
    """Take the most intense centroid within each configured reporter window."""
    mz_values = np.asarray(mz_values, dtype=float)
    intensity_values = np.asarray(intensity_values, dtype=float)
    if mz_values.ndim != 1 or intensity_values.ndim != 1 or len(mz_values) != len(intensity_values):
        raise ValueError("m/z and intensity arrays must be one-dimensional and aligned")
    if len(mz_values) and np.any(np.diff(mz_values) < 0):
        order = np.argsort(mz_values)
        mz_values, intensity_values = mz_values[order], intensity_values[order]
    records: list[dict[str, float | str]] = []
    for channel, expected in reporter_masses.items():
        left = np.searchsorted(mz_values, expected - tolerance_da, side="left")
        right = np.searchsorted(mz_values, expected + tolerance_da, side="right")
        if left == right:
            observed, intensity, mass_error = np.nan, 0.0, np.nan
        else:
            local = left + int(np.argmax(intensity_values[left:right]))
            observed = float(mz_values[local])
            intensity = float(intensity_values[local])
            mass_error = (observed - expected) / expected * 1_000_000.0
        records.append({
            "reporter_channel": channel,
            "reporter_mz_expected": float(expected),
            "reporter_mz_observed": observed,
            "reporter_intensity": intensity,
            "mass_error_ppm": mass_error,
        })
    return records


def _precursor_fields(spectrum: Any) -> tuple[float, float, float, str | None]:
    precursors = spectrum.selected_precursors
    if not precursors:
        return np.nan, np.nan, np.nan, None
    precursor = precursors[0]
    return (
        float(precursor.get("mz", np.nan)),
        float(precursor.get("charge", np.nan)),
        float(precursor.get("i", np.nan)),
        precursor.get("precursor id"),
    )


def extract_mzml_ms2(
    mzml_path: str | Path,
    raw_file_stem: str,
    reporter_masses: dict[str, float],
    tolerance_da: float,
    example_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[dict[str, Any]]]:
    """Parse one mzML and return spectrum and long reporter-ion tables."""
    scan_counts = {"total": 0, "ms1": 0, "ms2": 0, "other": 0}
    spectrum_records: list[dict[str, Any]] = []
    reporter_records: list[dict[str, Any]] = []
    examples: list[tuple[float, int, dict[str, Any]]] = []
    reader = pymzml.run.Reader(str(mzml_path), build_index_from_scratch=False)
    for spectrum in reader:
        scan_counts["total"] += 1
        level = int(spectrum.ms_level)
        if level == 1:
            scan_counts["ms1"] += 1
            continue
        if level != 2:
            scan_counts["other"] += 1
            continue
        scan_counts["ms2"] += 1
        scan_id = int(spectrum.ID)
        retention_time = float(spectrum.scan_time_in_minutes())
        precursor_mz, precursor_charge, precursor_intensity, precursor_id = _precursor_fields(spectrum)
        peaks = np.asarray(spectrum.peaks("centroided"), dtype=float)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            raise ValueError(f"unexpected centroid peak array for scan {scan_id}: {peaks.shape}")
        extracted = extract_reporters_from_peaks(
            peaks[:, 0], peaks[:, 1], reporter_masses, tolerance_da
        )
        reporter_sum = float(sum(row["reporter_intensity"] for row in extracted))
        matched_reporters = int(sum(row["reporter_intensity"] > 0 for row in extracted))
        spectrum_records.append({
            "raw_file": raw_file_stem,
            "scan_id": scan_id,
            "retention_time": retention_time,
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge,
            "precursor_intensity": precursor_intensity,
            "precursor_scan_id": precursor_id,
            "fragment_peak_count": int(len(peaks)),
            "fragment_total_intensity": float(peaks[:, 1].sum()),
            "fragment_base_peak_intensity": float(peaks[:, 1].max()) if len(peaks) else 0.0,
            "reporter_intensity_sum": reporter_sum,
            "matched_reporter_count": matched_reporters,
        })
        shared = {
            "raw_file": raw_file_stem,
            "scan_id": scan_id,
            "retention_time": retention_time,
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge,
        }
        reporter_records.extend({**shared, **row} for row in extracted)
        example = {
            "scan_id": scan_id,
            "retention_time": retention_time,
            "precursor_mz": precursor_mz,
            "mz": peaks[:, 0].copy(),
            "intensity": peaks[:, 1].copy(),
            "reporters": extracted,
        }
        candidate = (reporter_sum, scan_id, example)
        if len(examples) < example_count:
            heapq.heappush(examples, candidate)
        elif candidate[:2] > examples[0][:2]:
            heapq.heapreplace(examples, candidate)
    reporter_table = pd.DataFrame(reporter_records, columns=REPORTER_TABLE_COLUMNS)
    spectrum_table = pd.DataFrame(spectrum_records)
    selected_examples = [item[2] for item in sorted(examples, reverse=True)]
    return spectrum_table, reporter_table, scan_counts, selected_examples


def attach_channel_annotations(
    reporter_table: pd.DataFrame,
    manifest: pd.DataFrame,
    raw_file_stem: str,
    reporter_channels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join each extracted channel to the Stage 2 master manifest."""
    required = {
        "raw_file_stem", "reporter_channel", "sample_id", "label_raw",
        "sample_category", "is_valid_target", "exclusion_reason",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing channel annotation fields: {sorted(missing)}")
    run_mapping = manifest.loc[
        manifest["raw_file_stem"].eq(raw_file_stem),
        [
            "reporter_channel", "sample_id", "label_raw", "sample_category",
            "is_valid_target", "exclusion_reason", "is_single_cell", "is_carrier",
            "is_reference", "is_normalization", "is_empty", "is_unused",
        ],
    ].copy()
    if run_mapping.empty:
        raise ValueError(f"no manifest annotations found for {raw_file_stem}")
    if run_mapping["reporter_channel"].duplicated().any():
        raise ValueError(f"duplicate reporter annotations found for {raw_file_stem}")
    absent = sorted(set(reporter_channels) - set(run_mapping["reporter_channel"]))
    if absent:
        raise ValueError(f"reporter channels lack annotations: {absent}")
    joined = reporter_table.merge(
        run_mapping,
        on="reporter_channel",
        how="left",
        validate="many_to_one",
    )
    if joined["sample_category"].isna().any():
        raise AssertionError("one or more extracted reporters failed annotation mapping")
    return joined, run_mapping.sort_values(
        "reporter_channel", key=lambda values: values.str.extract(r"(\d+)")[0].astype(int)
    )


def reporter_channel_summary(reporter_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize peak recovery, signal, and mass accuracy for each channel."""
    grouped = reporter_table.groupby(
        [
            "reporter_channel", "reporter_mz_expected", "label_raw",
            "sample_category", "is_valid_target",
        ],
        dropna=False,
        sort=False,
    )
    summary = grouped.agg(
        ms2_scans=("scan_id", "nunique"),
        matched_scans=("reporter_mz_observed", "count"),
        total_reporter_intensity=("reporter_intensity", "sum"),
        median_reporter_intensity=("reporter_intensity", "median"),
        mean_reporter_intensity=("reporter_intensity", "mean"),
        reporter_intensity_standard_deviation=("reporter_intensity", "std"),
        median_mass_error_ppm=("mass_error_ppm", "median"),
        median_absolute_mass_error_ppm=("mass_error_ppm", lambda values: values.abs().median()),
    ).reset_index()
    summary["matched_scan_fraction"] = summary["matched_scans"] / summary["ms2_scans"]
    return summary


def validate_against_search_results(
    evidence_path: str | Path,
    raw_file_stem: str,
    reporter_table: pd.DataFrame,
    reporter_channels: list[str],
    annotation_channels: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare raw extracted intensities with MaxQuant's uncorrected reporters."""
    annotation_channels = annotation_channels or reporter_channels
    all_intensity_columns = [
        f"Reporter.intensity.{index}" for index in range(1, len(annotation_channels) + 1)
    ]
    intensity_columns = all_intensity_columns[:len(reporter_channels)]
    use_columns = ["Raw.file", "MS.MS.scan.number", *all_intensity_columns]
    evidence = pd.read_csv(evidence_path, usecols=use_columns, low_memory=False)
    evidence = evidence[evidence["Raw.file"].eq(raw_file_stem)].copy()
    if evidence.empty:
        raise ValueError(f"search-result table has no rows for {raw_file_stem}")
    evidence["MS.MS.scan.number"] = pd.to_numeric(
        evidence["MS.MS.scan.number"], errors="raise"
    ).astype(int)
    if evidence["MS.MS.scan.number"].duplicated().any():
        raise ValueError("pilot search-result table has duplicate MS2 scan numbers")
    populated_search_result_channels = [
        channel
        for channel, column in zip(annotation_channels, all_intensity_columns)
        if pd.to_numeric(evidence[column], errors="coerce").notna().any()
    ]
    long = evidence.melt(
        id_vars="MS.MS.scan.number",
        value_vars=intensity_columns,
        var_name="reporter_field",
        value_name="search_result_intensity",
    )
    field_to_channel = dict(zip(intensity_columns, reporter_channels))
    long["reporter_channel"] = long["reporter_field"].map(field_to_channel)
    long["search_result_intensity"] = pd.to_numeric(
        long["search_result_intensity"], errors="coerce"
    )
    comparison = reporter_table[[
        "scan_id", "reporter_channel", "reporter_intensity"
    ]].merge(
        long,
        left_on=["scan_id", "reporter_channel"],
        right_on=["MS.MS.scan.number", "reporter_channel"],
        how="inner",
        validate="one_to_one",
    )
    records = []
    for channel, frame in comparison.groupby("reporter_channel", sort=False):
        observed = frame.dropna(subset=["search_result_intensity"])
        if len(observed) > 1:
            log_correlation = float(
                np.corrcoef(
                    np.log1p(observed["reporter_intensity"]),
                    np.log1p(observed["search_result_intensity"]),
                )[0, 1]
            )
        else:
            log_correlation = np.nan
        nonzero = observed["search_result_intensity"].gt(0)
        ratios = (
            observed.loc[nonzero, "reporter_intensity"]
            / observed.loc[nonzero, "search_result_intensity"]
        )
        records.append({
            "reporter_channel": channel,
            "identified_psm_scans": int(len(frame)),
            "comparable_intensities": int(len(observed)),
            "nonzero_search_result_intensities": int(nonzero.sum()),
            "log1p_intensity_correlation": log_correlation,
            "median_extracted_to_search_result_ratio": float(ratios.median()) if len(ratios) else np.nan,
        })
    summary = pd.DataFrame(records)
    matched_scans = comparison["scan_id"].nunique()
    annotation_only_columns = all_intensity_columns[len(reporter_channels):]
    annotation_only_nonnull_values = int(sum(
        pd.to_numeric(evidence[column], errors="coerce").notna().sum()
        for column in annotation_only_columns
    ))
    return summary, {
        "identified_psm_rows": int(len(evidence)),
        "identified_psm_scans": int(evidence["MS.MS.scan.number"].nunique()),
        "identified_scans_matched_to_mzml": int(matched_scans),
        "populated_search_result_channels": populated_search_result_channels,
        "annotation_only_reporter_fields": len(annotation_only_columns),
        "annotation_only_nonnull_reporter_values": annotation_only_nonnull_values,
    }


def plot_example_spectra(
    examples: list[dict[str, Any]],
    reporter_masses: dict[str, float],
    output_path: Path,
) -> None:
    """Plot full fragment and magnified reporter regions for high-signal scans."""
    if not examples:
        raise ValueError("no MS2 spectra are available to plot")
    figure, axes = plt.subplots(
        len(examples), 2, figsize=(12, 3.0 * len(examples)), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(examples), 2)
    for row, example in enumerate(examples):
        mz_values = example["mz"]
        intensities = example["intensity"]
        normalized = intensities / intensities.max() * 100 if intensities.max() > 0 else intensities
        axes[row, 0].vlines(mz_values, 0, normalized, linewidth=0.55)
        axes[row, 0].set_xlim(max(100, float(mz_values.min())), min(1600, float(mz_values.max())))
        axes[row, 0].set_ylabel("Relative intensity (%)")
        axes[row, 0].set_title(
            f"Scan {example['scan_id']} | RT {example['retention_time']:.2f} min | "
            f"precursor {example['precursor_mz']:.4f} m/z"
        )
        window = (mz_values >= 125.7) & (mz_values <= 131.6)
        axes[row, 1].vlines(mz_values[window], 0, intensities[window], linewidth=0.8)
        reporter_y_max = axes[row, 1].get_ylim()[1]
        for channel_index, (channel, expected) in enumerate(reporter_masses.items()):
            axes[row, 1].axvline(expected, color="tab:red", linestyle="--", linewidth=0.65, alpha=0.65)
            label_height = 0.96 if channel_index % 2 == 0 else 0.77
            axes[row, 1].text(
                expected,
                reporter_y_max * label_height,
                channel,
                rotation=90,
                ha="right",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 0.4},
            )
        axes[row, 1].set_xlim(125.7, 131.6)
        axes[row, 1].set_title("TMT reporter-ion region")
        axes[row, 1].set_ylabel("Raw intensity")
    for axis in axes[-1]:
        axis.set_xlabel("m/z")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _format_channel_recognition(mapping: pd.DataFrame) -> str:
    records = []
    for category, frame in mapping.groupby("sample_category", sort=False):
        records.append(f"`{category}`: {', '.join(frame['reporter_channel'])}")
    return "; ".join(records)


def run_stage_ten(
    config_path: str | Path = "configs/spectrum_mil.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the complete single-file raw-spectrum feasibility pilot."""
    config = load_yaml_config(config_path)
    pilot = config["pilot"]
    raw_file_stem = str(pilot["raw_file_stem"])
    raw_path = Path(pilot["raw_path"])
    mzml_path = Path(pilot["mzml_path"])
    manifest_path = Path(pilot["manifest_path"])
    evidence_path = Path(pilot["evidence_path"])
    output_directory = Path(pilot.get("output_directory", "data/interim/spectra"))
    scheme_name = str(pilot["reporter_scheme"])
    reporter_masses = {
        str(channel): float(mass)
        for channel, mass in config["reporter_schemes"][scheme_name]["masses"].items()
    }
    tolerance_da = float(pilot["reporter_tolerance_da"])
    example_count = int(pilot["example_spectra_count"])
    for path in (raw_path, mzml_path, manifest_path, evidence_path):
        if not path.is_file():
            raise FileNotFoundError(f"Stage 10 input not found: {path}")
    scheme_validation = validate_reporter_scheme(reporter_masses, tolerance_da)
    header = inspect_mzml_header(mzml_path)
    if header.get("mzml_id") != raw_file_stem:
        raise ValueError(
            f"mzML ID {header.get('mzml_id')} does not match pilot {raw_file_stem}"
        )
    manifest = pd.read_parquet(manifest_path)
    annotated = manifest[manifest["raw_file_stem"].eq(raw_file_stem)]
    if annotated.empty:
        raise ValueError("pilot raw file is absent from the master manifest")
    if dry_run:
        return {
            "dry_run": True,
            "raw_file": raw_file_stem,
            "mzml_declared_spectra": int(header["declared_spectrum_count"]),
            "reporter_scheme": scheme_name,
            "extracted_channels": len(reporter_masses),
            "annotated_channels": int(annotated["reporter_channel"].nunique()),
            "valid_target_channels": int(annotated["is_valid_target"].sum()),
        }
    logger = logging.getLogger("single_cell_proteomics_mil")
    logger.info("Stage 10 | parsing %s", mzml_path)
    spectrum_table, reporter_table, scan_counts, examples = extract_mzml_ms2(
        mzml_path,
        raw_file_stem,
        reporter_masses,
        tolerance_da,
        example_count,
    )
    if scan_counts["total"] != int(header["declared_spectrum_count"]):
        raise AssertionError("parsed scan count differs from the mzML declaration")
    if scan_counts["ms1"] == 0 or scan_counts["ms2"] == 0:
        raise AssertionError("mzML must contain both MS1 and MS2 scans")
    if spectrum_table["precursor_mz"].isna().any():
        raise AssertionError("one or more MS2 scans lack precursor m/z")
    if (spectrum_table["fragment_peak_count"] == 0).any():
        raise AssertionError("one or more MS2 scans lack fragment peaks")
    reporter_table, channel_mapping = attach_channel_annotations(
        reporter_table,
        manifest,
        raw_file_stem,
        list(reporter_masses),
    )
    channel_summary = reporter_channel_summary(reporter_table)
    if channel_summary["total_reporter_intensity"].nunique() <= 1:
        raise AssertionError("reporter-channel sums are all zero or identical")
    if reporter_table["reporter_intensity"].gt(0).sum() == 0:
        raise AssertionError("no reporter-ion peaks were extracted")
    required_categories = {"target", "carrier", "normalization", "non_target"}
    if not required_categories.issubset(set(channel_mapping["sample_category"])):
        raise AssertionError("pilot mapping does not recognize all required channel categories")
    logger.info("Stage 10 | validating reporters against identified PSMs")
    psm_validation, psm_metadata = validate_against_search_results(
        evidence_path,
        raw_file_stem,
        reporter_table,
        list(reporter_masses),
        channel_mapping["reporter_channel"].tolist(),
    )
    median_psm_correlation = float(psm_validation["log1p_intensity_correlation"].median())
    median_psm_scale_ratio = float(
        psm_validation["median_extracted_to_search_result_ratio"].median()
    )
    if median_psm_correlation < 0.75:
        raise AssertionError("extracted reporter intensities disagree with search-result values")
    if psm_metadata["identified_scans_matched_to_mzml"] != psm_metadata["identified_psm_scans"]:
        raise AssertionError("one or more identified PSM scans are absent from the parsed mzML")
    if psm_metadata["annotation_only_nonnull_reporter_values"] != 0:
        raise AssertionError("annotation-only RI12-RI16 fields unexpectedly contain pilot values")
    table_directory = Path("outputs/tables")
    report_directory = Path("outputs/reports")
    figure_directory = Path("outputs/figures")
    for directory in (output_directory, table_directory, report_directory, figure_directory):
        directory.mkdir(parents=True, exist_ok=True)
    reporter_path = output_directory / f"stage_10_{raw_file_stem}_reporter_ions.parquet"
    spectra_path = output_directory / f"stage_10_{raw_file_stem}_ms2_spectra.parquet"
    reporter_table.to_parquet(reporter_path, index=False)
    spectrum_table.to_parquet(spectra_path, index=False)
    channel_summary.to_csv(table_directory / "stage_10_reporter_channel_summary.csv", index=False)
    channel_mapping.to_csv(table_directory / "stage_10_channel_mapping.csv", index=False)
    scheme_validation.to_csv(table_directory / "stage_10_reporter_scheme_validation.csv", index=False)
    psm_validation.to_csv(table_directory / "stage_10_psm_reporter_validation.csv", index=False)
    pd.DataFrame([scan_counts]).to_csv(table_directory / "stage_10_scan_summary.csv", index=False)
    plot_example_spectra(
        examples,
        reporter_masses,
        figure_directory / "stage_10_example_ms2_spectra.png",
    )
    embedded_algorithm = str(header.get("source_checksum_algorithm", "sha1"))
    logger.info("Stage 10 | calculating immutable input checksums")
    actual_embedded_checksum = file_checksum(raw_path, embedded_algorithm)
    header["source_checksum_actual"] = actual_embedded_checksum
    header["source_checksum_matches"] = actual_embedded_checksum == header.get("source_checksum_embedded")
    header["raw_sha256"] = file_checksum(raw_path, "sha256")
    header["mzml_sha256"] = file_checksum(mzml_path, "sha256")
    header["conversion_command"] = "not embedded in mzML"
    (table_directory / "stage_10_mzml_provenance.json").write_text(
        json.dumps(header, indent=2), encoding="utf-8"
    )
    if not header["source_checksum_matches"]:
        raise AssertionError("local RAW checksum differs from the RAW embedded in the mzML")
    charge_available = int(spectrum_table["precursor_charge"].notna().sum())
    metadata = {
        "raw_file": raw_file_stem,
        "reporter_scheme": scheme_name,
        "scheme_verification": config["reporter_schemes"][scheme_name]["verification"],
        "reporter_tolerance_da": tolerance_da,
        "total_scans": scan_counts["total"],
        "ms1_scans": scan_counts["ms1"],
        "ms2_scans": scan_counts["ms2"],
        "ms2_with_precursor_mz": int(spectrum_table["precursor_mz"].notna().sum()),
        "ms2_with_precursor_charge": charge_available,
        "reporter_measurements": int(len(reporter_table)),
        "reporter_peaks_matched": int(reporter_table["reporter_mz_observed"].notna().sum()),
        "annotated_channels": int(len(channel_mapping)),
        "extracted_channels": int(len(reporter_masses)),
        "valid_single_cell_channels": int(channel_mapping["is_valid_target"].sum()),
        "median_psm_log1p_intensity_correlation": median_psm_correlation,
        "median_extracted_to_search_result_ratio": median_psm_scale_ratio,
        "annotation_only_unused_channels": int(
            ((channel_mapping["is_unused"]) & (~channel_mapping["reporter_channel"].isin(reporter_masses))).sum()
        ),
        "raw_source_checksum_matches_mzml": bool(header["source_checksum_matches"]),
        "reporter_table": str(reporter_path),
        "spectrum_table": str(spectra_path),
        **psm_metadata,
    }
    (table_directory / "stage_10_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    unused_annotation = channel_mapping[
        ~channel_mapping["reporter_channel"].isin(reporter_masses)
    ]
    target_label_counts = (
        channel_mapping.loc[channel_mapping["is_valid_target"], "label_raw"]
        .value_counts()
        .sort_index()
    )
    report_lines = [
        "# Stage 10 — Raw-file conversion pilot",
        "",
        "## Pilot and provenance",
        "",
        f"- Pilot run: `{raw_file_stem}`.",
        f"- Existing immutable RAW was parsed through its converted mzML counterpart; the local RAW {embedded_algorithm.upper()} matches the source checksum embedded in mzML.",
        f"- Converter: `{header['software'][0]['id']}` version `{header['software'][0]['version']}`.",
        f"- Embedded processing: {', '.join(method for row in header['processing_methods'] for method in row['methods'])}.",
        "- The exact original conversion command is not embedded; tool/version, source path, processing methods, source checksum, and output checksum are recorded.",
        "",
        "## Parsed spectra",
        "",
        f"- Total scans: {scan_counts['total']:,} ({scan_counts['ms1']:,} MS1; {scan_counts['ms2']:,} MS2).",
        f"- MS2 scans with precursor m/z: {spectrum_table['precursor_mz'].notna().sum():,}/{scan_counts['ms2']:,}.",
        f"- MS2 scans with precursor charge: {charge_available:,}/{scan_counts['ms2']:,}.",
        f"- Every retained MS2 scan has centroided fragment peaks and retention time.",
        "",
        "## Reporter scheme and extraction",
        "",
        f"- Verified pilot scheme: `{scheme_name}` ({config['reporter_schemes'][scheme_name]['verification']}).",
        f"- Extraction used the configured monoisotopic masses and ±{tolerance_da:.3f} Da windows; all channel windows are non-overlapping.",
        f"- Extracted {len(reporter_table):,} scan-channel measurements ({scan_counts['ms2']:,} MS2 scans × {len(reporter_masses)} reporters).",
        f"- Nonzero matched reporter peaks: {reporter_table['reporter_mz_observed'].notna().sum():,}.",
        f"- Reporter channel sums vary (minimum {channel_summary['total_reporter_intensity'].min():,.0f}; maximum {channel_summary['total_reporter_intensity'].max():,.0f}), rather than being all zero or identical.",
        f"- Median per-channel log1p correlation with MaxQuant's uncorrected reporter values: {median_psm_correlation:.4f} across {psm_metadata['identified_psm_scans']:,} identified scans.",
        f"- The extracted-to-MaxQuant intensity ratio is consistently scaled (median {median_psm_scale_ratio:.3f} across channels); the validation therefore checks signal agreement, not exact equality of software-specific intensity units.",
        f"- Search-result channels with any populated values: {', '.join(psm_metadata['populated_search_result_channels'])}; RI12-RI16 contain zero populated pilot values.",
        "- RI12–RI16 are not extracted for this pilot: they are unified-annotation placeholders marked unused, not evidence of a TMTpro16 acquisition.",
        "",
        "## Channel mapping",
        "",
        f"- Recognized categories across the full annotation: {_format_channel_recognition(channel_mapping)}.",
        f"- Valid biological target channels: {channel_mapping['is_valid_target'].sum()} ({', '.join(f'{count} `{label}`' for label, count in target_label_counts.items())}).",
        f"- Annotation-only channels: {', '.join(unused_annotation['reporter_channel'])}; all are marked unused.",
        "- Carrier, normalization, non-target, and unused channels remain excluded from the binary target-cell corpus.",
        "",
        "## Stage gate",
        "",
        "Stage 10 passes: one converted mzML is provenance-linked to its RAW source, MS1/MS2 parsing works, TMT reporters vary and agree with the search-result extraction, and reporter channels map to the correct experimental annotations.",
        "",
    ]
    (report_directory / "stage_10_raw_pilot.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spectrum_mil.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging(Path("outputs/logs/stage_10_raw_pilot.log"))
    result = run_stage_ten(args.config, args.dry_run)
    logging.getLogger("single_cell_proteomics_mil").info("Stage 10 completed: %s", result)


if __name__ == "__main__":
    main()
